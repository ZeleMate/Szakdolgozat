"""
Module for reinforcement learning environment.
"""

import gym
from gym import spaces
import numpy as np
from ..models.embedding import encode_queries
from collections import deque

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity between the embeddings
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

class LegalSearchEnv(gym.Env):
    """
    Reinforcement learning environment for legal document search.
    """
    
    def __init__(self, model, documents, target_doc_idx=None):
        """
        Initialize environment.
        
        Args:
            model: SentenceTransformer model
            documents: List of document strings
            target_doc_idx: Index of the target document (ground truth)
        """
        super().__init__()
        self.model = model
        self.documents = documents
        self.action_space = spaces.Discrete(len(documents))  # choose a document
        
        # Sample embedding to get dimensions
        sample_embedding = encode_queries(model, ["test query"])[0]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(sample_embedding.shape[0],), 
            dtype=np.float32
        )
        
        # Query for the environment
        self.query_text = "Jogellenes elbocsátás miatt indított per"
        self.query = encode_queries(model, [self.query_text])[0]
        
        # Target document (simulated ground truth)
        self.target_doc_idx = target_doc_idx if target_doc_idx is not None else 1

    def reset(self):
        """Reset the environment and return initial observation."""
        return np.array(self.query, dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Document index to select
            
        Returns:
            observation, reward, done, info
        """
        # Reward is positive if correct document selected, negative otherwise
        reward = 1.0 if action == self.target_doc_idx else -0.5
        done = True  # one-step episode for now
        obs = np.array(self.query, dtype=np.float32)
        info = {"selected_doc": self.documents[action]}
        return obs, reward, done, info

class RewardModel:
    def __init__(self, embedding_dim: int = 3072):
        self.weights = np.zeros(embedding_dim)
        self.bias = 0.0
        self.learning_rate = 0.01
        self.feedback_history = deque(maxlen=1000)
    def predict_reward(self, embedding: np.ndarray) -> float:
        return np.dot(embedding, self.weights) + self.bias
    def update(self, embedding: np.ndarray, feedback: float):
        error = feedback - self.predict_reward(embedding)
        self.weights += self.learning_rate * error * embedding
        self.bias += self.learning_rate * error

class LegalSearchRLHF(gym.Env):
    def __init__(self, model, documents, search_engine, target_doc_idx=None):
        self.model = model
        self.documents = documents
        self.search_engine = search_engine
        self.target_doc_idx = target_doc_idx if target_doc_idx is not None else 1
        self.embedding_dim = encode_queries(model, ["test query"])[0].shape[0]
        self.reward_model = RewardModel(self.embedding_dim)
        self.doc_embeddings = [encode_queries(model, [doc])[0] for doc in documents]
        self.current_state = np.zeros(self.embedding_dim)
        self.steps_in_episode = 0
        self.max_steps_per_episode = 10
        self.action_space = spaces.Discrete(len(documents))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.embedding_dim,), dtype=np.float32)

    def step(self, action):
        selected_doc_embedding = self.doc_embeddings[action]
        target_similarity = compute_similarity(selected_doc_embedding, self.doc_embeddings[self.target_doc_idx])
        model_reward = self.reward_model.predict_reward(selected_doc_embedding)
        reward = 0.7 * target_similarity + 0.3 * model_reward
        self.current_state = (1 - 0.3) * self.current_state + 0.3 * selected_doc_embedding
        self.steps_in_episode += 1
        return self.current_state.astype(np.float32), reward, self.steps_in_episode >= self.max_steps_per_episode, {}

    def update_reward_model(self, doc_idx: int, feedback: float):
        self.reward_model.update(self.doc_embeddings[doc_idx], feedback)
"""
Main module for the Semantic Search + Reinforcement Learning project.
This module orchestrates the components of the system.
"""

import os
import sys

# Add the project root directory to sys.path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.search.semantic_search import SemanticSearch, GraphSemanticSearch
from src.rl.environment import LegalSearchEnv, LegalSearchRLHF
from src.models.embedding import load_embedding_model
from src.data_loader.legal_docs import load_documents_from_folder

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Semantic Search with RL")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base",
                      help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=5, 
                      help="Number of top documents to retrieve")
    parser.add_argument("--query", type=str, default="Jogellenes felmondás munkahelyen",
                      help="Search query")
    parser.add_argument("--doc_path", type=str, default="data/raw",
                      help="Path to folder containing legal documents")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                      help="Similarity threshold for graph-based search")
    parser.add_argument("--use_rl", action="store_true",
                      help="Use Reinforcement Learning for search refinement")
    return parser.parse_args()

def simulate_user_feedback(results, target_doc):
    """
    Simulate user feedback on search results.
    
    Args:
        results: List of (idx, doc) tuples from search results
        target_doc: The target document to compare against
        
    Returns:
        Dictionary mapping document indices to feedback scores
    """
    feedback = {}
    for idx, doc in results:
        # Simple simulation - higher score for longer common substrings
        common_text = len(set(doc.split()).intersection(set(target_doc.split()))) / len(set(doc.split()).union(set(target_doc.split())))
        feedback[idx] = min(1.0, max(0.1, common_text * 2))  # Scale between 0.1 and 1.0
    return feedback

def main():
    """Main function."""
    args = parse_args()

    # Load embedding model
    model = load_embedding_model(args.model)

    # Load real legal documents from folder
    documents = load_documents_from_folder(args.doc_path)
    if not documents:
        print("No documents found in folder:", args.doc_path)
        return

    # Initialize semantic search
    semantic_search = GraphSemanticSearch(model, documents, similarity_threshold=args.similarity_threshold)

    # Search query
    results = semantic_search.search(args.query, args.top_k)
    print(f"Query: {args.query}")
    print("Top documents:")
    for i, doc in results:
        print(f"- {doc[:200]}...\n")  # print preview

    # Initialize RL environment
    env = LegalSearchRLHF(model, documents, semantic_search, target_doc_idx=1)

    # Simple random agent for demonstration
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print("\nRL Demo:")
    print(f"Selected document: {documents[action][:200]}...")
    print(f"Reward: {reward}")

    if args.use_rl:
        for idx, score in simulate_user_feedback(results, documents[env.target_doc_idx]).items():
            env.update_reward_model(idx, score)
            semantic_search.update_weights(idx, score)
        refined_results = semantic_search.search(args.query, top_k=args.top_k, use_pagerank=True)
        print("Refined top documents:")
        for i, doc in refined_results:
            print(f"- {doc[:200]}...\n")  # print preview

if __name__ == "__main__":
    main()
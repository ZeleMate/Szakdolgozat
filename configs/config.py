"""
Configuration settings for the project.
"""

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384  # Depends on the model

# Search configuration
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# RL configuration
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPISODES = 1000

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models"
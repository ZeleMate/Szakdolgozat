"""
Module for embedding models and vector operations.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    """Base class for embedding models"""
    
    def encode(self, texts):
        """Encode texts into embedding vectors"""
        raise NotImplementedError("Subclasses must implement encode method")

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for Sentence Transformer models"""
    
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        """Encode texts into embedding vectors"""
        return self.model.encode(texts)

class BertEmbeddingModel(EmbeddingModel):
    """Wrapper for BERT-based embedding models"""
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, texts):
        """Encode texts into embedding vectors"""
        # Mean Pooling - Take attention mask into account for averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Tokenize and prepare inputs
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.numpy()

def load_embedding_model(model_name):
    """Load an embedding model based on the model name.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        EmbeddingModel: An embedding model instance
    """
    if model_name == "MODERNBert":
        # Hungarian BERT model
        return BertEmbeddingModel("SZTAKI-HLT/hubert-base-cc")
    elif model_name == "distiluse-base-multilingual-cased-v1":
        # Multilingual SentenceTransformer model
        return SentenceTransformerModel(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
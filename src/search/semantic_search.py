"""
Module for semantic search functionality.
"""

import numpy as np
import faiss
import networkx as nx
from ..models.embedding import encode_documents, encode_queries

def compute_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

class SemanticSearch:
    """Class for semantic search functionality."""
    
    def __init__(self, model, documents):
        """
        Initialize semantic search with documents.
        
        Args:
            model: SentenceTransformer model
            documents: List of document strings
        """
        self.model = model
        self.documents = documents
        self.doc_embeddings = encode_documents(model, documents)
        self.similarity_threshold = 0.7
        self._create_index()
        self._build_document_graph()
    
    def _create_index(self):
        """Create FAISS index from document embeddings."""
        self.index = faiss.IndexFlatL2(self.doc_embeddings[0].shape[0])
        self.index.add(np.array(self.doc_embeddings).astype('float32'))
    
    def _build_document_graph(self):
        self.document_graph = nx.Graph()
        for i, _ in enumerate(self.documents):
            self.document_graph.add_node(i, text=self.documents[i][:100] + "...")
        for i in range(len(self.documents)):
            for j in range(i + 1, len(self.documents)):
                similarity = compute_similarity(self.doc_embeddings[i], self.doc_embeddings[j])
                if similarity >= self.similarity_threshold:
                    self.document_graph.add_edge(i, j, weight=similarity)
        self.pagerank_scores = nx.pagerank(self.document_graph, weight='weight')
    
    def search(self, query, top_k=5, use_pagerank=True, expand_results=True):
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_pagerank: Whether to use PageRank scores in ranking
            expand_results: Whether to expand results using graph neighbors
            
        Returns:
            List of (index, document) tuples
        """
        query_embedding = encode_queries(self.model, [query])
        dists, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            top_k * 2 if expand_results else top_k
        )
        initial_results = [(int(i), self.documents[int(i)], float(d)) for d, i in zip(dists[0], indices[0])]
        if expand_results:
            expanded_indices = set(idx for idx, _, _ in initial_results)
            for idx, _, _ in initial_results[:3]:
                expanded_indices.update(self.document_graph.neighbors(idx))
            candidates = [(idx, self.documents[idx], compute_similarity(query_embedding[0], self.doc_embeddings[idx]) * 0.8 + self.pagerank_scores.get(idx, 0) * 0.2) for idx in expanded_indices]
            return sorted(candidates, key=lambda x: x[2], reverse=True)[:top_k]
        return initial_results[:top_k]

class GraphSemanticSearch(SemanticSearch):
    """
    Enhanced semantic search using graph-based algorithms.
    Extends SemanticSearch with graph analysis capabilities.
    """
    
    def __init__(self, model, documents, similarity_threshold=0.7):
        """
        Initialize graph-enhanced semantic search with documents.
        
        Args:
            model: SentenceTransformer model
            documents: List of document strings
            similarity_threshold: Threshold for connecting documents in the graph
        """
        self.similarity_threshold = similarity_threshold
        self.document_weights = {}  # Store weights for personalized PageRank
        super().__init__(model, documents)
    
    def _build_document_graph(self):
        """
        Build document similarity graph with additional metrics.
        """
        self.document_graph = nx.Graph()
        
        # Add nodes
        for i, doc in enumerate(self.documents):
            self.document_graph.add_node(i, text=self.documents[i][:100] + "...")
            self.document_weights[i] = 1.0  # Default weight
        
        # Add edges based on similarity
        for i in range(len(self.documents)):
            for j in range(i + 1, len(self.documents)):
                similarity = compute_similarity(self.doc_embeddings[i], self.doc_embeddings[j])
                if similarity >= self.similarity_threshold:
                    self.document_graph.add_edge(i, j, weight=similarity)
        
        # Calculate PageRank
        self._update_pagerank()
    
    def _update_pagerank(self):
        """Update PageRank scores using current weights."""
        personalization = {i: weight for i, weight in self.document_weights.items()}
        self.pagerank_scores = nx.pagerank(
            self.document_graph, 
            weight='weight',
            personalization=personalization
        )
    
    def update_weights(self, doc_idx, score):
        """
        Update document weights based on feedback.
        
        Args:
            doc_idx: Index of document to update
            score: Feedback score (higher is better)
        """
        if doc_idx in self.document_weights:
            # Blend new score with existing weight
            self.document_weights[doc_idx] = 0.7 * self.document_weights[doc_idx] + 0.3 * score
            # Update PageRank with new weights
            self._update_pagerank()
    
    def search(self, query, top_k=5, use_pagerank=True, expand_results=True):
        """
        Enhanced search with graph-based re-ranking.
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_pagerank: Whether to use PageRank scores in ranking
            expand_results: Whether to expand results using graph neighbors
            
        Returns:
            List of (index, document) tuples
        """
        query_embedding = encode_queries(self.model, [query])
        dists, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            top_k * 2 if expand_results else top_k
        )
        
        initial_results = [(int(i), self.documents[int(i)], float(d)) for d, i in zip(dists[0], indices[0])]
        
        # If we're not using graph features, just return basic results
        if not use_pagerank and not expand_results:
            return [(idx, doc) for idx, doc, _ in initial_results[:top_k]]
        
        # Enhance results using graph
        expanded_indices = set(idx for idx, _, _ in initial_results)
        
        # Add neighbors from the graph to expand results
        if expand_results:
            for idx, _, _ in initial_results[:3]:  # Use top 3 initial results
                if idx in self.document_graph:
                    expanded_indices.update(self.document_graph.neighbors(idx))
        
        # Calculate scores using both similarity and PageRank
        candidates = []
        for idx in expanded_indices:
            sim_score = compute_similarity(query_embedding[0], self.doc_embeddings[idx])
            pagerank_boost = self.pagerank_scores.get(idx, 0)
            final_score = sim_score * (0.8 if use_pagerank else 1.0) + pagerank_boost * (0.2 if use_pagerank else 0.0)
            candidates.append((idx, self.documents[idx], final_score))
        
        # Sort by final score and return top_k
        return [(idx, doc) for idx, doc, _ in sorted(candidates, key=lambda x: x[2], reverse=True)[:top_k]]
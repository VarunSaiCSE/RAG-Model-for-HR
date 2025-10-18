# File: src/vector_store.py
"""
FAISS vector store operations with incremental updates
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.utils import log_message, EMBEDDINGS_DIR


class VectorStore:
    """FAISS-based vector store with persistence"""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize vector store"""
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []  # List of dicts: {candidate_id, candidate_name, source, text}
        
        self.index_path = os.path.join(EMBEDDINGS_DIR, "faiss_index.index")
        self.metadata_path = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
    
    def create_index(self):
        """Create new FAISS index"""
        log_message("Creating new FAISS index")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings and metadata to index
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: list of metadata dicts for each embedding
        """
        if self.index is None:
            self.create_index()
        
        if len(embeddings) == 0:
            return
        
        # Ensure correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Add to metadata
        self.metadata.extend(metadata)
        
        log_message(f"Added {len(embeddings)} embeddings to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            log_message("Index is empty", "WARNING")
            return []
        
        # Reshape query
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)
                results.append((self.metadata[idx], similarity))
        
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        if self.index is None or self.index.ntotal == 0:
            log_message("Nothing to save", "WARNING")
            return
        
        log_message(f"Saving index with {self.index.ntotal} vectors")
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        log_message("Vector store saved successfully")
    
    def load(self) -> bool:
        """Load index and metadata from disk"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            log_message("No existing vector store found", "INFO")
            return False
        
        try:
            log_message("Loading existing vector store")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            log_message(f"Loaded vector store with {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            log_message(f"Error loading vector store: {str(e)}", "ERROR")
            return False
    
    def merge(self, other_embeddings: np.ndarray, other_metadata: List[Dict]):
        """Merge new embeddings into existing index"""
        if self.index is None:
            self.create_index()
        
        self.add(other_embeddings, other_metadata)

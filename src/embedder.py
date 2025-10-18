# File: src/embedder.py
"""
Text embedding generation using sentence-transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.utils import log_message


class Embedder:
    """Text embedding generator"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize embedding model"""
        log_message(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        log_message(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        log_message(f"Encoding {len(texts)} text chunks...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        log_message(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string"""
        return self.model.encode([text], convert_to_numpy=True, 
                                normalize_embeddings=True)[0]

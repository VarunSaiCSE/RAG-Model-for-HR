# File: src/retriever.py
"""
Retrieval module for finding relevant document chunks
"""

from typing import List, Dict, Tuple
from src.vector_store import VectorStore
from src.embedder import Embedder
from src.utils import log_message, TOP_K, SIMILARITY_THRESHOLD


class Retriever:
    """Document retriever using vector similarity search"""
    
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """Initialize retriever"""
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = TOP_K, 
                 threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            top_k: Number of results
            threshold: Minimum similarity threshold
        
        Returns:
            List of metadata dicts with similarity scores
        """
        log_message(f"Retrieving documents for query: {query[:100]}...")
        
        # Embed query
        query_embedding = self.embedder.encode_single(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Filter by threshold and deduplicate
        filtered_results = []
        seen_texts = set()
        
        for metadata, similarity in results:
            if similarity >= threshold and metadata['text'] not in seen_texts:
                metadata['similarity'] = similarity
                filtered_results.append(metadata)
                seen_texts.add(metadata['text'])
            
            if len(filtered_results) >= top_k:
                break
        
        log_message(f"Retrieved {len(filtered_results)} relevant chunks")
        return filtered_results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents as context string"""
        if not retrieved_docs:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            candidate_name = doc.get('candidate_name', 'Unknown')
            source = doc.get('source', 'Unknown')
            text = doc.get('text', '')
            
            context_parts.append(
                f"[Source {i}] Candidate: {candidate_name} | Source: {source}\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)

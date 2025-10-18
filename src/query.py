# File: src/query.py
"""
Query handler - orchestrates retrieval and generation
"""

from typing import Dict, List
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_model import RAGModel
from src.utils import log_message, TOP_K


class QueryHandler:
    """Handles user queries end-to-end"""
    
    def __init__(self):
        """Initialize query handler"""
        log_message("Initializing Query Handler...")
        
        # Load components
        self.embedder = Embedder()
        self.vector_store = VectorStore(embedding_dim=self.embedder.embedding_dim)
        
        # Load existing vector store
        if not self.vector_store.load():
            raise RuntimeError(
                "No vector store found. Please run ingestion first:\n"
                "python main.py --mode ingest --source csv"
            )
        
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.rag_model = RAGModel()
        
        log_message("Query Handler initialized successfully")
    
    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        Process a query and return answer with sources
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        log_message(f"\n{'='*80}")
        log_message(f"QUERY: {question}")
        log_message(f"{'='*80}\n")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information in the candidate database.",
                'sources': []
            }
        
        # Format context
        context = self.retriever.format_context(retrieved_docs)
        
        # Generate answer
        answer = self.rag_model.generate(question, context)
        
        # Extract sources
        sources = self._extract_sources(retrieved_docs)
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Extract and format source information"""
        sources = []
        seen = set()
        
        for doc in retrieved_docs:
            candidate_name = doc.get('candidate_name', 'Unknown')
            source = doc.get('source', 'Unknown')
            similarity = doc.get('similarity', 0.0)
            
            key = f"{candidate_name}|{source}"
            if key not in seen:
                sources.append({
                    'candidate': candidate_name,
                    'source': source,
                    'relevance': f"{similarity:.2%}"
                })
                seen.add(key)
        
        return sources
    
    def format_output(self, result: Dict) -> str:
        """Format query result for display"""
        output = []
        output.append("\n" + "="*80)
        output.append("ANSWER")
        output.append("="*80)
        output.append(result['answer'])
        
        if result['sources']:
            output.append("\n" + "="*80)
            output.append("SOURCES")
            output.append("="*80)
            for i, source in enumerate(result['sources'], 1):
                output.append(
                    f"{i}. Candidate: {source['candidate']} | "
                    f"Source: {source['source']} | "
                    f"Relevance: {source['relevance']}"
                )
        
        output.append("="*80 + "\n")
        return "\n".join(output)

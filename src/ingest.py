# File: src/ingest.py
"""
Data ingestion orchestrator - fetches and processes all candidate data
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict
from tqdm import tqdm
import os

from src.utils import (
    log_message, clean_text, split_into_chunks, get_candidate_id,
    load_processed_ids, save_processed_ids, DATA_DIR, CREDENTIALS_DIR, SHEET_NAME
)
from src.pdf_parser import parse_resume
from src.linkedin_scraper import parse_linkedin
from src.github_scraper import parse_github
from src.embedder import Embedder
from src.vector_store import VectorStore


class DataIngestion:
    """Handles all data ingestion and processing"""
    
    def __init__(self, incremental: bool = False):
        """
        Initialize data ingestion
        
        Args:
            incremental: If True, only process new/modified candidates
        """
        self.incremental = incremental
        self.embedder = Embedder()
        self.vector_store = VectorStore(embedding_dim=self.embedder.embedding_dim)
        
        # Load existing data if incremental
        if self.incremental:
            self.vector_store.load()
            self.processed_ids = load_processed_ids()
        else:
            self.processed_ids = {}
    
    def load_from_csv(self, csv_path: str = None) -> pd.DataFrame:
        """Load candidate data from CSV"""
        if csv_path is None:
            csv_path = os.path.join(DATA_DIR, "sheets.csv")
        
        log_message(f"Loading data from CSV: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        log_message(f"Loaded {len(df)} candidates from CSV")
        return df
    
    def load_from_gsheet(self, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
        """Load candidate data from Google Sheets"""
        log_message(f"Loading data from Google Sheet: {sheet_name}")
        
        try:
            # Setup credentials
            creds_path = os.path.join(CREDENTIALS_DIR, "service_account.json")
            
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Service account credentials not found at {creds_path}\n"
                    "Please download credentials from Google Cloud Console"
                )
            
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
            client = gspread.authorize(creds)
            
            # Open sheet
            sheet = client.open(sheet_name).sheet1
            data = sheet.get_all_records()
            
            df = pd.DataFrame(data)
            log_message(f"Loaded {len(df)} candidates from Google Sheets")
            return df
        
        except Exception as e:
            log_message(f"Error loading from Google Sheets: {str(e)}", "ERROR")
            raise
    
    def process_candidate(self, row: Dict) -> List[Dict]:
        """
        Process single candidate - fetch all sources and create chunks
        
        Args:
            row: Candidate data row
        
        Returns:
            List of chunk metadata dicts
        """
        candidate_name = row.get('Name', 'Unknown')
        candidate_id = get_candidate_id(row)
        
        # Skip if already processed and incremental mode
        if self.incremental and candidate_id in self.processed_ids:
            return []
        
        log_message(f"\n{'='*60}")
        log_message(f"Processing candidate: {candidate_name}")
        log_message(f"{'='*60}")
        
        # Collect all text sources
        all_text = []
        sources = []
        
        # Resume
        resume_url = row.get('Resume_URL', '')
        if resume_url:
            resume_text = parse_resume(resume_url, candidate_name)
            if resume_text:
                all_text.append(resume_text)
                sources.append('Resume')
        
        # LinkedIn
        linkedin_url = row.get('LinkedIn_URL', '')
        if linkedin_url:
            linkedin_text = parse_linkedin(linkedin_url, candidate_name)
            if linkedin_text:
                all_text.append(linkedin_text)
                sources.append('LinkedIn')
        
        # GitHub
        github_url = row.get('GitHub_URL', '')
        if github_url:
            github_text = parse_github(github_url, candidate_name)
            if github_text:
                all_text.append(github_text)
                sources.append('GitHub')
        
        # Combine and clean
        combined_text = "\n\n".join(all_text)
        combined_text = clean_text(combined_text)
        
        if not combined_text:
            log_message(f"No text extracted for {candidate_name}", "WARNING")
            return []
        
        log_message(f"Total extracted text: {len(combined_text)} characters")
        
        # Split into chunks
        chunks = split_into_chunks(combined_text)
        log_message(f"Created {len(chunks)} chunks")
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk_text in enumerate(chunks):
            metadata = {
                'candidate_id': candidate_id,
                'candidate_name': candidate_name,
                'source': ', '.join(sources),
                'chunk_id': i,
                'text': chunk_text
            }
            chunk_metadata.append(metadata)
        
        return chunk_metadata
    
    def ingest(self, source: str = 'csv', csv_path: str = None, 
               sheet_name: str = SHEET_NAME):
        """
        Main ingestion function
        
        Args:
            source: 'csv' or 'gsheet'
            csv_path: Path to CSV file (if source='csv')
            sheet_name: Google Sheet name (if source='gsheet')
        """
        log_message("\n" + "="*80)
        log_message("STARTING DATA INGESTION")
        log_message("="*80)
        
        # Load data
        if source == 'csv':
            df = self.load_from_csv(csv_path)
        elif source == 'gsheet':
            df = self.load_from_gsheet(sheet_name)
        else:
            raise ValueError(f"Invalid source: {source}. Use 'csv' or 'gsheet'")
        
        # Process each candidate
        all_chunks = []
        all_embeddings = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing candidates"):
            try:
                chunk_metadata = self.process_candidate(row.to_dict())
                
                if chunk_metadata:
                    # Extract texts for embedding
                    chunk_texts = [meta['text'] for meta in chunk_metadata]
                    
                    # Generate embeddings
                    embeddings = self.embedder.encode(chunk_texts, show_progress=False)
                    
                    all_chunks.extend(chunk_metadata)
                    all_embeddings.append(embeddings)
                    
                    # Mark as processed
                    candidate_id = get_candidate_id(row.to_dict())
                    from datetime import datetime
                    self.processed_ids[candidate_id] = datetime.now().isoformat()
            
            except Exception as e:
                log_message(f"Error processing candidate {row.get('Name', 'Unknown')}: {str(e)}", "ERROR")
                continue
        
        # Combine all embeddings
        if all_embeddings:
            import numpy as np
            all_embeddings = np.vstack(all_embeddings)
            
            # Add to vector store
            log_message(f"\n{'='*60}")
            log_message(f"Adding {len(all_chunks)} chunks to vector store")
            self.vector_store.add(all_embeddings, all_chunks)
            
            # Save vector store
            self.vector_store.save()
            
            # Save processed IDs
            save_processed_ids(self.processed_ids)
            
            log_message("\n" + "="*80)
            log_message("INGESTION COMPLETE")
            log_message(f"Total chunks indexed: {self.vector_store.index.ntotal}")
            log_message("="*80)
        else:
            log_message("No data to ingest", "WARNING")

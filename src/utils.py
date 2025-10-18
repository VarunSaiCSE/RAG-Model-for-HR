# File: src/utils.py
"""
Utility functions and configuration constants
"""

import os
import json
#import re
from datetime import datetime
from typing import Dict, List, Any

# Configuration
CHUNK_SIZE = 500  # Words per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score
MAX_NEW_TOKENS = 300  # Max length of generated answer
TEMPERATURE = 0.7  # LLM temperature
SHEET_NAME = "HR Candidates"  # Google Sheet name

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESUMES_DIR = os.path.join(DATA_DIR, "resumes")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CREDENTIALS_DIR = os.path.join(BASE_DIR, "credentials")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESUMES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(CREDENTIALS_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                     overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= 10:  # Minimum 10 words per chunk
            chunks.append(chunk)
    
    return chunks


def convert_drive_url(url: str) -> str:
    """Convert Google Drive share URL to direct download URL"""
    if not url or 'drive.google.com' not in url:
        return url
    
    # Extract file ID from various Google Drive URL formats
    file_id = None
    
    if '/file/d/' in url:
        file_id = url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        file_id = url.split('id=')[1].split('&')[0]
    
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return url


def load_processed_ids(filepath: str = None) -> Dict[str, str]:
    """Load dictionary of processed candidate IDs and timestamps"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, "processed_ids.json")
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_processed_ids(processed_ids: Dict[str, str], filepath: str = None):
    """Save processed candidate IDs and timestamps"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, "processed_ids.json")
    
    with open(filepath, 'w') as f:
        json.dump(processed_ids, f, indent=2)


def get_candidate_id(row: Dict[str, Any]) -> str:
    """Generate unique candidate ID from row data"""
    # Use email as primary ID, fallback to name
    return row.get('Email', '').lower().strip() or row.get('Name', '').lower().strip()


def log_message(message: str, level: str = "INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


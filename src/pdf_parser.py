# File: src/pdf_parser.py
"""
PDF text extraction using PyPDF2
"""

import os
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from typing import Optional
from src.utils import log_message, RESUMES_DIR, convert_drive_url


def download_pdf(url: str, save_path: str) -> bool:
    """Download PDF from URL"""
    try:
        # Convert Google Drive URL to direct download
        url = convert_drive_url(url)
        
        # Download with timeout
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Check if response is PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
            log_message(f"URL does not point to PDF: {url}", "WARNING")
            return False
        
        # Save PDF
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return True
    
    except Exception as e:
        log_message(f"Error downloading PDF from {url}: {str(e)}", "ERROR")
        return False


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract text from local PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
            except Exception as e:
                log_message(f"Error extracting page {page_num + 1}: {str(e)}", "WARNING")
                continue
        
        return text.strip() if text else None
    
    except Exception as e:
        log_message(f"Error reading PDF {pdf_path}: {str(e)}", "ERROR")
        return None


def extract_text_from_url(url: str, candidate_name: str) -> Optional[str]:
    """Download and extract text from PDF URL"""
    if not url:
        return None
    
    try:
        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in candidate_name)
        pdf_filename = f"{safe_name}.pdf"
        pdf_path = os.path.join(RESUMES_DIR, pdf_filename)
        
        # Download if not already exists
        if not os.path.exists(pdf_path):
            log_message(f"Downloading resume for {candidate_name}")
            if not download_pdf(url, pdf_path):
                return None
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            log_message(f"Successfully extracted {len(text)} chars from {candidate_name}'s resume")
        else:
            log_message(f"No text extracted from {candidate_name}'s resume", "WARNING")
        
        return text
    
    except Exception as e:
        log_message(f"Error processing resume for {candidate_name}: {str(e)}", "ERROR")
        return None


def parse_resume(url: str, candidate_name: str) -> str:
    """Main function to parse resume and return text"""
    if not url or url.lower() in ['', 'none', 'n/a', 'null']:
        log_message(f"No resume URL for {candidate_name}", "INFO")
        return ""
    
    text = extract_text_from_url(url, candidate_name)
    return text if text else ""

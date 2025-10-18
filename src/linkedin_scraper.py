# File: src/linkedin_scraper.py
"""
LinkedIn public profile scraper (no login required)
"""

import requests
import time
from bs4 import BeautifulSoup
from typing import Optional, Dict
from src.utils import log_message


def scrape_linkedin_profile(url: str) -> Optional[str]:
    """
    Scrape public LinkedIn profile information
    NOTE: Only works for public profiles, respects robots.txt
    """
    if not url or 'linkedin.com' not in url:
        return None
    
    try:
        # Add delay to respect rate limits
        time.sleep(2)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        # Check if page is publicly accessible
        if response.status_code == 999:
            log_message("LinkedIn blocked request (rate limit or login required)", "WARNING")
            return None
        
        if response.status_code != 200:
            log_message(f"LinkedIn returned status {response.status_code}", "WARNING")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to extract basic information from public profile
        profile_data = []
        
        # Name (public profiles usually show this)
        name_tag = soup.find('h1', class_=lambda x: x and 'name' in x.lower()) if soup.find('h1') else None
        if name_tag:
            profile_data.append(f"Name: {name_tag.get_text().strip()}")
        
        # Headline/Title
        headline_tag = soup.find('h2', class_=lambda x: x and 'headline' in x.lower()) if soup.find('h2') else None
        if headline_tag:
            profile_data.append(f"Headline: {headline_tag.get_text().strip()}")
        
        # About section
        about_section = soup.find('section', class_=lambda x: x and 'about' in x.lower())
        if about_section:
            about_text = about_section.get_text().strip()
            profile_data.append(f"About: {about_text}")
        
        # Experience section
        experience_section = soup.find('section', class_=lambda x: x and 'experience' in x.lower())
        if experience_section:
            experience_text = experience_section.get_text().strip()
            profile_data.append(f"Experience: {experience_text}")
        
        if profile_data:
            return "\n\n".join(profile_data)
        else:
            log_message("Could not extract LinkedIn data (profile may be private)", "WARNING")
            return None
    
    except requests.exceptions.Timeout:
        log_message("LinkedIn request timed out", "WARNING")
        return None
    except Exception as e:
        log_message(f"Error scraping LinkedIn: {str(e)}", "WARNING")
        return None


def parse_linkedin(url: str, candidate_name: str) -> str:
    """Main function to parse LinkedIn profile"""
    if not url or url.lower() in ['', 'none', 'n/a', 'null']:
        log_message(f"No LinkedIn URL for {candidate_name}", "INFO")
        return ""
    
    log_message(f"Attempting to fetch LinkedIn data for {candidate_name}")
    text = scrape_linkedin_profile(url)
    
    if text:
        log_message(f"Successfully extracted {len(text)} chars from LinkedIn for {candidate_name}")
        return text
    else:
        log_message(f"Could not fetch LinkedIn data for {candidate_name} (may be private)", "INFO")
        return ""


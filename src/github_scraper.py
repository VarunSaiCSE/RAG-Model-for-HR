# File: src/github_scraper.py
"""
GitHub public profile and repository scraper
"""

import requests
import time
from bs4 import BeautifulSoup
from typing import Optional, List
from src.utils import log_message


def scrape_github_profile(username: str) -> Optional[str]:
    """Scrape GitHub profile information"""
    try:
        time.sleep(1)  # Rate limiting
        
        url = f"https://github.com/{username}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        profile_data = []
        
        # Bio
        bio_div = soup.find('div', class_=lambda x: x and 'user-profile-bio' in str(x))
        if bio_div:
            profile_data.append(f"Bio: {bio_div.get_text().strip()}")
        
        # Stats
        stats = soup.find_all('span', class_='Counter')
        if stats:
            profile_data.append(f"GitHub Stats: {' '.join([s.get_text().strip() for s in stats])}")
        
        return "\n".join(profile_data) if profile_data else None
    
    except Exception as e:
        log_message(f"Error scraping GitHub profile: {str(e)}", "WARNING")
        return None


def get_user_repos(username: str) -> List[Dict]:
    """Get user's public repositories using GitHub API"""
    try:
        time.sleep(1)
        
        url = f"https://api.github.com/users/{username}/repos"
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
        
        repos = response.json()
        return repos[:10]  # Limit to top 10 repos
    
    except Exception as e:
        log_message(f"Error fetching GitHub repos: {str(e)}", "WARNING")
        return []


def get_repo_readme(username: str, repo_name: str) -> Optional[str]:
    """Fetch repository README content"""
    try:
        time.sleep(1)
        
        # Try to get README.md
        url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/README.md"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            # Try master branch
            url = f"https://raw.githubusercontent.com/{username}/{repo_name}/master/README.md"
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.text
        
        return None
    
    except Exception as e:
        return None


def scrape_github_data(url: str) -> Optional[str]:
    """Main GitHub scraping function"""
    if not url or 'github.com' not in url:
        return None
    
    try:
        # Extract username from URL
        username = url.rstrip('/').split('/')[-1]
        if not username:
            return None
        
        all_data = []
        
        # Profile data
        profile_info = scrape_github_profile(username)
        if profile_info:
            all_data.append(f"=== GitHub Profile ===\n{profile_info}")
        
        # Repository data
        repos = get_user_repos(username)
        if repos:
            repo_texts = []
            for repo in repos:
                repo_name = repo.get('name', '')
                description = repo.get('description', '')
                language = repo.get('language', '')
                stars = repo.get('stargazers_count', 0)
                
                repo_text = f"\nRepo: {repo_name}"
                if description:
                    repo_text += f"\nDescription: {description}"
                if language:
                    repo_text += f"\nLanguage: {language}"
                if stars > 0:
                    repo_text += f"\nStars: {stars}"
                
                # Try to get README for top repos
                if stars > 5 or len(repo_texts) < 3:
                    readme = get_repo_readme(username, repo_name)
                    if readme:
                        # Limit README length
                        readme_preview = readme[:500] + "..." if len(readme) > 500 else readme
                        repo_text += f"\nREADME: {readme_preview}"
                
                repo_texts.append(repo_text)
            
            if repo_texts:
                all_data.append(f"\n=== GitHub Repositories ===\n" + "\n---\n".join(repo_texts))
        
        return "\n\n".join(all_data) if all_data else None
    
    except Exception as e:
        log_message(f"Error processing GitHub data: {str(e)}", "WARNING")
        return None


def parse_github(url: str, candidate_name: str) -> str:
    """Main function to parse GitHub profile"""
    if not url or url.lower() in ['', 'none', 'n/a', 'null']:
        log_message(f"No GitHub URL for {candidate_name}", "INFO")
        return ""
    
    log_message(f"Fetching GitHub data for {candidate_name}")
    text = scrape_github_data(url)
    
    if text:
        log_message(f"Successfully extracted {len(text)} chars from GitHub for {candidate_name}")
        return text
    else:
        log_message(f"Could not fetch GitHub data for {candidate_name}", "INFO")
        return ""

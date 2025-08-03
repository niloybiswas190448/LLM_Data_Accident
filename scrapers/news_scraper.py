"""
News Scraper Module for Road Accident Analysis Pipeline
Handles scraping of multiple Bangladeshi news sources
"""

import requests
import time
import random
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from datetime import datetime, timedelta
import re

from config import NEWS_SOURCES, SCRAPING_CONFIG, ACCIDENT_KEYWORDS
from utils.logger import get_pipeline_logger, log_execution_time

class NewsScraper:
    """
    Scraper class for extracting accident-related news from Bangladeshi sources
    """
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self.session = requests.Session()
        self.ua = UserAgent()
        self.setup_session()
    
    def setup_session(self):
        """Set up the requests session with headers and retry strategy"""
        self.session.headers.update({
            'User-Agent': SCRAPING_CONFIG["user_agent"],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def get_random_delay(self) -> float:
        """Get a random delay between requests to avoid being blocked"""
        return random.uniform(SCRAPING_CONFIG["delay_between_requests"], 
                             SCRAPING_CONFIG["delay_between_requests"] * 2)
    
    def make_request(self, url: str, retries: int = None) -> Optional[requests.Response]:
        """
        Make a request with retry logic and error handling
        
        Args:
            url: URL to request
            retries: Number of retry attempts
            
        Returns:
            Response object or None if failed
        """
        if retries is None:
            retries = SCRAPING_CONFIG["retry_attempts"]
        
        for attempt in range(retries + 1):
            try:
                # Add random delay
                time.sleep(self.get_random_delay())
                
                # Update user agent randomly
                self.session.headers['User-Agent'] = self.ua.random
                
                response = self.session.get(
                    url, 
                    timeout=SCRAPING_CONFIG["timeout"],
                    allow_redirects=True
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{retries + 1}): {url} - {str(e)}")
                if attempt == retries:
                    self.logger.error(f"Failed to fetch {url} after {retries + 1} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def is_accident_related(self, text: str, language: str) -> bool:
        """
        Check if text contains accident-related keywords
        
        Args:
            text: Text to check
            language: Language of the text ('bangla' or 'english')
            
        Returns:
            True if accident-related, False otherwise
        """
        text_lower = text.lower()
        keywords = ACCIDENT_KEYWORDS.get(language, [])
        
        # Check for keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # Require at least 2 keyword matches for confidence
        return keyword_matches >= 2
    
    def extract_article_links(self, soup: BeautifulSoup, source_config: Dict) -> List[str]:
        """
        Extract article links from a search results page
        
        Args:
            soup: BeautifulSoup object of the page
            source_config: Configuration for the news source
            
        Returns:
            List of article URLs
        """
        links = []
        selectors = source_config["selectors"]["article_links"]
        
        for selector in selectors.split(", "):
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    # Make relative URLs absolute
                    if not href.startswith('http'):
                        href = urljoin(source_config["base_url"], href)
                    links.append(href)
        
        return list(set(links))  # Remove duplicates
    
    def extract_article_content(self, soup: BeautifulSoup, source_config: Dict) -> Dict:
        """
        Extract article content from a news page
        
        Args:
            soup: BeautifulSoup object of the article page
            source_config: Configuration for the news source
            
        Returns:
            Dictionary containing extracted content
        """
        content = {
            'title': '',
            'text': '',
            'date': '',
            'url': ''
        }
        
        # Extract title
        title_selectors = source_config["selectors"]["title"]
        for selector in title_selectors.split(", "):
            title_elem = soup.select_one(selector)
            if title_elem:
                content['title'] = title_elem.get_text(strip=True)
                break
        
        # Extract main content
        content_selectors = source_config["selectors"]["content"]
        for selector in content_selectors.split(", "):
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script and style elements
                for script in content_elem(["script", "style"]):
                    script.decompose()
                content['text'] = content_elem.get_text(strip=True)
                break
        
        # Extract date
        date_selectors = source_config["selectors"]["date"]
        for selector in date_selectors.split(", "):
            date_elem = soup.select_one(selector)
            if date_elem:
                content['date'] = date_elem.get_text(strip=True)
                break
        
        return content
    
    def parse_date(self, date_str: str, language: str) -> Optional[datetime]:
        """
        Parse date string to datetime object
        
        Args:
            date_str: Date string to parse
            language: Language of the date string
            
        Returns:
            Parsed datetime object or None
        """
        if not date_str:
            return None
        
        # Common date patterns
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
            r'(\d{1,2})\s+(জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)\s+(\d{4})',  # Bangla months
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if 'জানুয়ারি' in pattern:  # Bangla pattern
                        day, month_bn, year = match.groups()
                        month_map = {
                            'জানুয়ারি': 1, 'ফেব্রুয়ারি': 2, 'মার্চ': 3, 'এপ্রিল': 4,
                            'মে': 5, 'জুন': 6, 'জুলাই': 7, 'আগস্ট': 8,
                            'সেপ্টেম্বর': 9, 'অক্টোবর': 10, 'নভেম্বর': 11, 'ডিসেম্বর': 12
                        }
                        month = month_map.get(month_bn, 1)
                        return datetime(int(year), month, int(day))
                    else:
                        groups = match.groups()
                        if len(groups[0]) == 4:  # YYYY/MM/DD
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                        else:  # DD/MM/YYYY
                            return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @log_execution_time
    def scrape_source(self, source_name: str, keywords: List[str] = None) -> List[Dict]:
        """
        Scrape accident-related articles from a specific news source
        
        Args:
            source_name: Name of the news source
            keywords: List of keywords to search for
            
        Returns:
            List of extracted articles
        """
        if source_name not in NEWS_SOURCES:
            self.logger.error(f"Unknown news source: {source_name}")
            return []
        
        source_config = NEWS_SOURCES[source_name]
        language = source_config["language"]
        
        if keywords is None:
            keywords = ACCIDENT_KEYWORDS[language]
        
        articles = []
        processed_urls = set()
        
        self.logger.info(f"Starting to scrape {source_name}")
        
        for keyword in keywords[:5]:  # Limit to first 5 keywords to avoid too many requests
            try:
                # Construct search URL
                search_url = source_config["search_url"].format(query=keyword)
                self.logger.info(f"Searching for keyword: {keyword}")
                
                # Get search results page
                response = self.make_request(search_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article links
                article_links = self.extract_article_links(soup, source_config)
                self.logger.info(f"Found {len(article_links)} potential articles for keyword '{keyword}'")
                
                # Process each article
                for link in article_links[:SCRAPING_CONFIG["max_articles_per_source"] // len(keywords)]:
                    if link in processed_urls:
                        continue
                    
                    processed_urls.add(link)
                    
                    # Get article page
                    article_response = self.make_request(link)
                    if not article_response:
                        continue
                    
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract article content
                    article_content = self.extract_article_content(article_soup, source_config)
                    article_content['url'] = link
                    article_content['source'] = source_name
                    article_content['language'] = language
                    
                    # Check if article is accident-related
                    if self.is_accident_related(article_content['title'] + ' ' + article_content['text'], language):
                        # Parse date
                        article_content['parsed_date'] = self.parse_date(article_content['date'], language)
                        
                        articles.append(article_content)
                        self.logger.debug(f"Extracted accident article: {article_content['title'][:50]}...")
                    
                    # Respect rate limits
                    time.sleep(self.get_random_delay())
                
            except Exception as e:
                self.logger.error(f"Error scraping {source_name} for keyword '{keyword}': {str(e)}")
                continue
        
        self.logger.info(f"Completed scraping {source_name}. Found {len(articles)} accident-related articles")
        return articles
    
    @log_execution_time
    def scrape_all_sources(self) -> List[Dict]:
        """
        Scrape all configured news sources
        
        Returns:
            List of all extracted articles
        """
        all_articles = []
        
        for source_name in NEWS_SOURCES.keys():
            try:
                articles = self.scrape_source(source_name)
                all_articles.extend(articles)
            except Exception as e:
                self.logger.error(f"Failed to scrape {source_name}: {str(e)}")
                continue
        
        self.logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles
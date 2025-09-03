import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import logging
import re
from urllib.parse import urljoin, urlparse
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    Handles web scraping and content extraction from URLs
    Uses BeautifulSoup for HTML parsing and content cleaning
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 30
    
    def scrape_url(self, url: str) -> Dict:
        """
        Scrape content from a URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL format")
            
            # Fetch the webpage
            response = self._fetch_page(url)
            
            if not response:
                raise ValueError("Failed to fetch webpage")
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            title = self._extract_title(soup)
            content = self._extract_main_content(soup)
            metadata = self._extract_metadata(soup, url)
            
            # Clean content
            content = self._clean_content(content)
            
            if not content or len(content.strip()) < 50:
                raise ValueError("No meaningful content extracted from webpage")
            
            return {
                "title": title,
                "content": content,
                "url": url,
                "metadata": metadata,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            raise
    
    def _fetch_page(self, url: str) -> Optional[requests.Response]:
        """
        Fetch webpage with error handling
        
        Args:
            url: URL to fetch
            
        Returns:
            Response object or None if failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                raise ValueError(f"URL does not contain HTML content: {content_type}")
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise ValueError(f"Failed to fetch webpage: {str(e)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract page title
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Page title
        """
        # Try different title selectors
        title_selectors = [
            'title',
            'h1',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    title = element.get('content', '').strip()
                else:
                    title = element.get_text().strip()
                
                if title and len(title) > 3:
                    return title
        
        return "Untitled"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from webpage
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Main content text
        """
        # Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Try different content selectors
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '.main-content'
        ]
        
        content = ""
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    content += self._get_clean_text(element) + "\n"
                break
        
        # If no specific content found, try to get body content
        if not content.strip():
            body = soup.find('body')
            if body:
                content = self._get_clean_text(body)
        
        return content
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """
        Remove unwanted HTML elements
        
        Args:
            soup: BeautifulSoup object
        """
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer',
            '.advertisement', '.ads', '.ad', '.sidebar',
            '.navigation', '.menu', '.social', '.share',
            '.comments', '.comment', '.related', '.tags',
            '.breadcrumb', '.pagination', '.cookie-notice'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _get_clean_text(self, element) -> str:
        """
        Extract clean text from HTML element
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Clean text content
        """
        if not element:
            return ""
        
        # Get text and clean it
        text = element.get_text()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize extracted content
        
        Args:
            content: Raw extracted content
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove excessive whitespace and normalize
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)  # Remove standalone numbers
        content = re.sub(r'^[^\w\s]*$', '', content, flags=re.MULTILINE)  # Remove lines with only symbols
        
        # Split into paragraphs and clean
        paragraphs = content.split('\n')
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 10:  # Only keep meaningful paragraphs
                cleaned_paragraphs.append(paragraph)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Extract metadata from webpage
        
        Args:
            soup: BeautifulSoup object
            url: Original URL
            
        Returns:
            Dict with metadata
        """
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc,
            "description": "",
            "keywords": [],
            "author": "",
            "published_date": ""
        }
        
        # Extract description
        desc_selectors = [
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]'
        ]
        
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                metadata["description"] = element.get('content', '').strip()
                break
        
        # Extract keywords
        keywords_element = soup.select_one('meta[name="keywords"]')
        if keywords_element:
            keywords = keywords_element.get('content', '')
            metadata["keywords"] = [k.strip() for k in keywords.split(',') if k.strip()]
        
        # Extract author
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author',
            '.byline'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    metadata["author"] = element.get('content', '').strip()
                else:
                    metadata["author"] = element.get_text().strip()
                break
        
        return metadata
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_clean_content(self, soup: BeautifulSoup) -> str:
        """
        Public method to get clean content from BeautifulSoup object
        (for external use)
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Clean text content
        """
        return self._extract_main_content(soup)



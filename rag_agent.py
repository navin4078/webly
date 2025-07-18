import os
import sys
from typing import List, TypedDict, Annotated, Optional, Dict, Set, Iterator
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import csv
import requests  # Added missing import
from urllib.parse import urljoin, urlparse, urlencode
from urllib.robotparser import RobotFileParser
import time
import random
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

# LangChain imports - UPDATED FOR CURRENT VERSION
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.base import Embeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# LangGraph imports for modern conversation memory
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Sentence Transformers import
from sentence_transformers import SentenceTransformer

# Advanced web scraping imports
from bs4 import BeautifulSoup, Comment
import re
from fake_useragent import UserAgent
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ScrapingConfig:
    """Advanced scraping configuration"""
    max_depth: int = 2
    max_pages: int = 50
    max_concurrent: int = 10
    delay_range: tuple = (1, 3)
    timeout: int = 30
    retries: int = 3
    respect_robots: bool = False  # Changed to False to bypass robots.txt by default
    use_sitemap: bool = True
    content_types: List[str] = None
    exclude_patterns: List[str] = None
    custom_headers: Dict[str, str] = None
    output_formats: List[str] = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = ['text/html', 'application/xhtml+xml']
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                r'.*\.(pdf|jpg|jpeg|png|gif|svg|mp4|mp3|zip|exe|doc|docx)$',
                r'.*(login|register|cart|checkout|admin|api).*',
                r'.*\#.*'  # Skip anchors
            ]
        if self.custom_headers is None:
            self.custom_headers = {}
        if self.output_formats is None:
            self.output_formats = ['json']

@dataclass 
class ScrapedPage:
    """Data structure for scraped page information"""
    url: str
    title: str = ""
    content: str = ""
    metadata: Dict = None
    links: List[str] = None
    scrape_time: float = 0
    status_code: int = 0
    content_type: str = ""
    depth: int = 0
    parent_url: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.links is None:
            self.links = []

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses"""
    
    def __init__(self):
        self.tokens = []
        self.current_response = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        self.tokens.append(token)
        self.current_response += token
        
    def reset(self):
        """Reset the handler for a new response"""
        self.tokens = []
        self.current_response = ""

class AdvancedUserAgentManager:
    """Advanced user agent rotation with realistic patterns"""
    
    def __init__(self):
        try:
            self.ua = UserAgent()
            self.custom_agents = [
                # Chrome variants
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                # Firefox variants
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
                # Safari variants
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                # Edge variants
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
            ]
        except Exception:
            self.ua = None
            
    def get_random_user_agent(self) -> str:
        """Get a random realistic user agent"""
        try:
            if self.ua and random.choice([True, False]):
                return self.ua.random
            return random.choice(self.custom_agents)
        except Exception:
            return self.custom_agents[0]
    
    def get_realistic_headers(self) -> Dict[str, str]:
        """Generate realistic browser headers (without brotli to avoid compression issues)"""
        return {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',  # Removed 'br' (brotli) to avoid decoding issues
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

class SentenceTransformerEmbeddings(Embeddings):
    """
    Custom LangChain-compatible embeddings using SentenceTransformers
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embeddings
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        print(f"🔍 Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ SentenceTransformer model loaded successfully!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a list of floats
        """
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

# Custom State for our RAG agent with conversation memory
class RAGState(MessagesState):
    """Extended state that includes conversation context and RAG-specific data"""
    context: str = ""  # Retrieved context from vector store
    last_query: str = ""  # Last user query for context

class WebScraperRAGAgentWithMemory:
    def __init__(self):
        """
        Initialize the Advanced Web Scraper RAG agent with conversation memory and streaming
        """
        # Get Google API key from environment
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        
        # Set environment variable for Google AI
        os.environ["GOOGLE_API_KEY"] = self.google_api_key
        
        # Initialize SentenceTransformer embeddings
        print("🚀 Initializing SentenceTransformer embeddings...")
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Gemini LLM (streaming handled separately)
        print("🤖 Initializing Gemini 2.5 Flash LLM...")
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.3
            # Note: streaming is handled in our custom implementation
        )
        
        # Initialize streaming callback handler
        self.streaming_handler = StreamingCallbackHandler()
        
        # Advanced scraping components
        self.ua_manager = AdvancedUserAgentManager()
        self.scraped_pages: List[ScrapedPage] = []
        self.failed_urls: Set[str] = set()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.session_stats = {
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'total_time': 0,
            'avg_page_time': 0
        }
        
        # RAG components (unchanged)
        self.vector_store = None
        self.history_aware_retriever = None
        self.question_answer_chain = None
        self.rag_chain = None
        self.graph = None
        self.checkpointer = MemorySaver()
        self.scraped_urls = []
        
        print("✅ Advanced WebScraperRAGAgentWithMemory initialized successfully!")
        
    async def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt
        
        Note: This method is only used when respect_robots=True in ScrapingConfig.
        When respect_robots=False, this check is bypassed entirely.
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(base_url, '/robots.txt')
            
            if base_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self.robots_cache[base_url] = rp
            
            return self.robots_cache[base_url].can_fetch('*', url)
        except Exception:
            return True  # Allow if can't parse robots.txt
    
    async def discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover URLs from sitemap.xml"""
        sitemap_urls = []
        try:
            parsed_url = urlparse(base_url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Common sitemap locations
            sitemap_locations = [
                '/sitemap.xml',
                '/sitemap_index.xml',
                '/sitemaps.xml',
                '/sitemap/sitemap.xml'
            ]
            
            async with aiohttp.ClientSession() as session:
                for location in sitemap_locations:
                    sitemap_url = urljoin(base_domain, location)
                    try:
                        async with session.get(sitemap_url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                sitemap_urls.extend(self._parse_sitemap(content, base_domain))
                                break
                    except Exception:
                        continue
        except Exception as e:
            logging.warning(f"Failed to discover sitemap: {e}")
        
        return sitemap_urls[:100]  # Limit sitemap URLs
    
    def _parse_sitemap(self, content: str, base_domain: str) -> List[str]:
        """Parse sitemap XML and extract URLs"""
        urls = []
        try:
            root = ET.fromstring(content)
            
            # Handle sitemap index
            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None:
                    urls.append(loc.text)
            
            # Handle URL entries
            for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None:
                    urls.append(loc.text)
                    
        except Exception as e:
            logging.warning(f"Failed to parse sitemap: {e}")
        
        return [url for url in urls if url and url.startswith(('http://', 'https://'))]
    
    async def fetch_page_async(self, session: aiohttp.ClientSession, url: str, 
                              semaphore: asyncio.Semaphore, config: ScrapingConfig,
                              depth: int = 0, parent_url: str = "") -> Optional[ScrapedPage]:
        """Asynchronously fetch and process a single page with advanced features"""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Check robots.txt if enabled
                if config.respect_robots:
                    if not await self.check_robots_txt(url):
                        logging.info(f"🚫 Robots.txt disallows: {url}")
                        return None
                else:
                    logging.debug(f"⚠️ Bypassing robots.txt for: {url}")
                
                # Skip excluded patterns
                if any(re.match(pattern, url, re.IGNORECASE) for pattern in config.exclude_patterns):
                    logging.info(f"⏭️ Skipping excluded URL: {url}")
                    return None
                
                # Prepare headers (avoid brotli compression)
                headers = self.ua_manager.get_realistic_headers()
                # Override Accept-Encoding to avoid brotli issues
                headers['Accept-Encoding'] = 'gzip, deflate'
                headers.update(config.custom_headers)
                
                # Add random delay for rate limiting
                if config.delay_range:
                    delay = random.uniform(*config.delay_range)
                    await asyncio.sleep(delay)
                
                # Make request with retries
                for attempt in range(config.retries + 1):
                    try:
                        timeout = aiohttp.ClientTimeout(total=config.timeout)
                        async with session.get(url, headers=headers, timeout=timeout) as response:
                            # Check content type
                            content_type = response.headers.get('content-type', '').lower()
                            if not any(ct in content_type for ct in config.content_types):
                                logging.info(f"⏭️ Skipping non-HTML content: {url}")
                                return None
                            
                            # Check status code
                            if response.status != 200:
                                logging.warning(f"⚠️ HTTP {response.status} for {url}")
                                if response.status >= 400:
                                    return None
                            
                            # Get content
                            content = await response.text()
                            
                            # Create scraped page object
                            page = ScrapedPage(
                                url=url,
                                content=content,
                                status_code=response.status,
                                content_type=content_type,
                                depth=depth,
                                parent_url=parent_url,
                                scrape_time=time.time() - start_time
                            )
                            
                            # Parse content with BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract title
                            title_tag = soup.find('title')
                            page.title = title_tag.get_text().strip() if title_tag else ""
                            
                            # Clean and extract text content
                            page.content = self._advanced_content_cleaning(soup)
                            
                            # Extract metadata
                            page.metadata = self._extract_metadata(soup, response.headers)
                            
                            # Extract links for further crawling
                            page.links = self._extract_links(soup, url)
                            
                            logging.info(f"✅ Successfully scraped: {url} ({len(page.content)} chars)")
                            return page
                            
                    except asyncio.TimeoutError:
                        logging.warning(f"⏰ Timeout on attempt {attempt + 1} for {url}")
                        if attempt < config.retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    except Exception as e:
                        logging.warning(f"⚠️ Error on attempt {attempt + 1} for {url}: {e}")
                        if attempt < config.retries:
                            await asyncio.sleep(2 ** attempt)
                        continue
                
                logging.error(f"❌ Failed to scrape after {config.retries + 1} attempts: {url}")
                self.failed_urls.add(url)
                return None
                
            except Exception as e:
                logging.error(f"❌ Critical error scraping {url}: {e}")
                self.failed_urls.add(url)
                return None
    
    def _advanced_content_cleaning(self, soup: BeautifulSoup) -> str:
        """Advanced content cleaning and extraction"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'noscript', 'iframe', 'object', 'embed']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract main content areas
        main_content = ""
        content_selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '.main-content', '.post-content',
            '.entry-content', '.article-content', '#content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text() for elem in elements])
                break
        
        # If no main content found, get all text
        if not main_content:
            main_content = soup.get_text()
        
        # Advanced text cleaning
        main_content = re.sub(r'\n\s*\n+', '\n\n', main_content)  # Multiple newlines
        main_content = re.sub(r'[ \t]+', ' ', main_content)       # Multiple spaces/tabs
        main_content = re.sub(r'\r\n?', '\n', main_content)       # Normalize line endings
        
        # Remove very short lines (likely navigation/footer elements)
        lines = main_content.split('\n')
        cleaned_lines = [line.strip() for line in lines 
                        if len(line.strip()) > 20 or line.strip() in ['', ' ']]
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_metadata(self, soup: BeautifulSoup, headers: Dict) -> Dict:
        """Extract comprehensive metadata from page"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Open Graph data
        og_data = {}
        for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            og_data[meta.get('property')] = meta.get('content')
        if og_data:
            metadata['open_graph'] = og_data
        
        # Schema.org structured data
        schema_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                schema_data.append(json.loads(script.string))
            except:
                pass
        if schema_data:
            metadata['schema_org'] = schema_data
        
        # Response headers
        metadata['content_length'] = headers.get('content-length', 0)
        metadata['last_modified'] = headers.get('last-modified', '')
        metadata['server'] = headers.get('server', '')
        
        return metadata
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links from page"""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Only include links from same domain
            if parsed.netloc == base_domain:
                # Clean URL
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean_url += f"?{parsed.query}"
                
                links.append(clean_url)
        
        return list(set(links))  # Remove duplicates
    
    async def advanced_crawl_website(self, url: str, config: ScrapingConfig) -> List[ScrapedPage]:
        """Advanced asynchronous website crawling with modern techniques"""
        print(f"🌐 Starting advanced crawl of: {url}")
        print(f"📊 Config: {config.max_pages} pages, {config.max_concurrent} concurrent, depth {config.max_depth}")
        robots_status = "respecting" if config.respect_robots else "bypassing"
        print(f"🤖 Robots.txt: {robots_status}")
        
        start_time = time.time()
        scraped_pages = []
        urls_to_process = [(url, 0, "")]  # (url, depth, parent_url)
        processed_urls = set()
        
        # Discover URLs from sitemap if enabled
        if config.use_sitemap:
            print("🗺️ Discovering URLs from sitemap...")
            sitemap_urls = await self.discover_sitemap_urls(url)
            if sitemap_urls:
                print(f"📍 Found {len(sitemap_urls)} URLs in sitemap")
                urls_to_process.extend([(u, 0, url) for u in sitemap_urls[:config.max_pages//2]])
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Create session with advanced configuration
        connector = aiohttp.TCPConnector(
            limit=config.max_concurrent * 2,
            limit_per_host=config.max_concurrent,
            enable_cleanup_closed=True
        )
        
        # Create session with explicit headers and timeout (no brotli compression)
        timeout = aiohttp.ClientTimeout(total=config.timeout, connect=10)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Accept-Encoding': 'gzip, deflate',  # Remove brotli from default encoding
                'User-Agent': 'Mozilla/5.0 (compatible; RAG-Scraper/1.0)'
            }
        ) as session:
            current_depth = 0
            
            while urls_to_process and len(scraped_pages) < config.max_pages and current_depth <= config.max_depth:
                # Process current depth level
                current_batch = [(u, d, p) for u, d, p in urls_to_process if d == current_depth]
                urls_to_process = [(u, d, p) for u, d, p in urls_to_process if d > current_depth]
                
                if not current_batch:
                    current_depth += 1
                    continue
                
                print(f"📂 Processing depth {current_depth}: {len(current_batch)} URLs")
                
                # Create tasks for current batch
                tasks = []
                for page_url, depth, parent_url in current_batch:
                    if page_url not in processed_urls and len(scraped_pages) < config.max_pages:
                        task = self.fetch_page_async(session, page_url, semaphore, config, depth, parent_url)
                        tasks.append(task)
                        processed_urls.add(page_url)
                
                # Execute tasks concurrently
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, ScrapedPage):
                            scraped_pages.append(result)
                            
                            # Add discovered links for next depth level
                            if result.depth < config.max_depth:
                                for link in result.links:
                                    if link not in processed_urls:
                                        urls_to_process.append((link, result.depth + 1, result.url))
                
                # Progress update
                print(f"✅ Completed depth {current_depth}: {len(scraped_pages)} pages scraped")
                current_depth += 1
        
        # Update session stats
        total_time = time.time() - start_time
        self.session_stats.update({
            'total_pages': len(processed_urls),
            'successful_pages': len(scraped_pages),
            'failed_pages': len(self.failed_urls),
            'total_time': total_time,
            'avg_page_time': total_time / max(len(scraped_pages), 1)
        })
        
        print(f"🎉 Crawling completed!")
        print(f"📊 Stats: {len(scraped_pages)} successful, {len(self.failed_urls)} failed, {total_time:.2f}s total")
        
        return scraped_pages
    
    def save_crawl_results(self, scraped_pages: List[ScrapedPage], output_dir: str = "./crawl_results"):
        """Save crawling results in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        if 'json' in getattr(self, 'config', ScrapingConfig()).output_formats:
            json_data = {
                'metadata': {
                    'crawl_time': time.time(),
                    'stats': self.session_stats,
                    'total_pages': len(scraped_pages)
                },
                'pages': [asdict(page) for page in scraped_pages]
            }
            
            with open(output_path / 'crawl_results.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        if 'csv' in getattr(self, 'config', ScrapingConfig()).output_formats:
            with open(output_path / 'crawl_results.csv', 'w', newline='', encoding='utf-8') as f:
                if scraped_pages:
                    writer = csv.DictWriter(f, fieldnames=asdict(scraped_pages[0]).keys())
                    writer.writeheader()
                    for page in scraped_pages:
                        row = asdict(page)
                        # Convert complex fields to strings
                        row['metadata'] = json.dumps(row['metadata'])
                        row['links'] = json.dumps(row['links'])
                        writer.writerow(row)
        
        print(f"💾 Results saved to {output_path}")
    
    def scrape_website(self, url: str, max_depth: int = 1, max_pages: int = 10, respect_robots: bool = False):
        """Legacy method wrapper for backward compatibility"""
        config = ScrapingConfig(
            max_depth=max_depth,
            max_pages=max_pages,
            max_concurrent=min(10, max_pages),
            respect_robots=respect_robots  # Allow explicit control over robots.txt
        )
        
        # Run async crawling
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            scraped_pages = loop.run_until_complete(
                self.advanced_crawl_website(url, config)
            )
            
            # Convert to LangChain documents for compatibility
            documents = []
            for page in scraped_pages:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=page.content,
                    metadata={
                        'source': page.url,
                        'title': page.title,
                        'scrape_time': page.scrape_time,
                        'depth': page.depth,
                        **page.metadata
                    }
                )
                documents.append(doc)
            
            self.scraped_pages = scraped_pages
            self.scraped_urls = [page.url for page in scraped_pages]
            
            # Save results
            self.save_crawl_results(scraped_pages)
            
            return documents
            
        finally:
            loop.close()
    
    def _clean_content(self, content: str) -> str:
        """Clean scraped content"""
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        content = content.strip()
        
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
    
    def _extract_links_simple(self, base_url: str, headers: dict) -> List[str]:
        """Extract internal links from a webpage for crawling (simple version)"""
        try:
            response = requests.get(base_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            base_domain = urlparse(base_url).netloc
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                if urlparse(full_url).netloc == base_domain:
                    if not any(skip in full_url.lower() for skip in 
                             ['#', 'javascript:', 'mailto:', '.pdf', '.jpg', '.png', 
                              'login', 'register', 'cart', 'checkout']):
                        links.append(full_url)
            
            return list(set(links))[:20]
            
        except Exception as e:
            print(f"⚠️ Failed to extract links from {base_url}: {e}")
            return []
    
    def load_and_process_website(self, url: str, crawl_depth: int = 1, max_pages: int = 10, 
                                auto_save_path: str = "./vector_store", 
                                advanced_config: Optional[ScrapingConfig] = None,
                                respect_robots: bool = False):
        """
        Load and process website content with advanced crawling and conversation memory setup
        
        Args:
            url: URL to scrape
            crawl_depth: How deep to crawl (1 = just main page, 2 = main + linked pages)
            max_pages: Maximum pages to scrape
            auto_save_path: Path where to automatically save the vector store
            advanced_config: Optional advanced scraping configuration
            respect_robots: Whether to respect robots.txt (False by default to bypass robots.txt)
        """
        print(f"🌐 Loading website with advanced scraping: {url}")
        print(f"📊 Crawl depth: {crawl_depth}, Max pages: {max_pages}")
        robots_status = "respecting" if respect_robots else "bypassing"
        print(f"🤖 Robots.txt: {robots_status}")
        
        # Use advanced config if provided, otherwise create default
        if advanced_config is None:
            advanced_config = ScrapingConfig(
                max_depth=crawl_depth,
                max_pages=max_pages,
                max_concurrent=min(10, max_pages),
                delay_range=(0.5, 2.0),
                respect_robots=respect_robots,  # Use the parameter value
                use_sitemap=True
            )
        
        # Store config for later use
        self.config = advanced_config
        
        # Scrape the website using advanced crawler
        documents = self.scrape_website(url, crawl_depth, max_pages)
        
        if not documents:
            raise ValueError("No content could be scraped from the website. Please check the URL.")
        
        print(f"📝 Scraped {len(documents)} pages")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"📄 Split into {len(texts)} text chunks")
        
        # Create vector store
        print("🔍 Creating embeddings and FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        # Save vector store
        print(f"💾 Saving vector store to: {auto_save_path}")
        self.save_vector_store(auto_save_path)
        
        # Create conversation memory-enabled RAG chain
        self._create_conversational_rag_chain()
        
        # Create LangGraph with conversation memory
        self._create_langgraph_agent()
        
        print("🎉 RAG agent with conversation memory is ready!")
        
    def _create_conversational_rag_chain(self):
        """
        Create a conversation-aware RAG chain using LangChain's history-aware retriever
        """
        print("🔗 Creating conversation-aware RAG chain...")
        
        # 1. Create a history-aware retriever
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vector_store.as_retriever(search_kwargs={"k": 6}), contextualize_q_prompt
        )
        
        # 2. Enhanced question-answering chain with citations for source extraction
        qa_system_prompt = """You are an expert assistant for question-answering tasks based on website content. \
Use the following pieces of retrieved context to answer the question comprehensively and accurately.

IMPORTANT CITATION REQUIREMENTS:
- ALWAYS include citations for any claims or information you provide
- Use the format [Source: page_title](source_url) for each citation
- Include multiple citations when referencing different sources
- Place citations immediately after the relevant information
- If you don't know the answer based on the provided context, say so clearly

FORMATTING REQUIREMENTS:
- Structure your response with clear sections using ### headings when appropriate
- Use **bold text** for important points and key terms
- Use *italic text* for emphasis
- Use numbered lists (1. 2. 3.) for step-by-step information
- Use bullet points (•) for general lists
- Use `code formatting` for technical terms, URLs, or specific values
- Include relevant quotes using > blockquotes when helpful
- Ensure proper spacing between sections

RESPONSE STRUCTURE:
1. Provide a direct answer to the question
2. Include supporting details with proper citations
3. Add context or background information if relevant
4. Conclude with a summary if the response is long

Remember: Every factual claim must have a citation with the source URL!

Context pieces with sources:
{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. Create the full RAG chain
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, 
            self.question_answer_chain
        )
        
        print("✅ Enhanced conversation-aware RAG chain with citations created!")
    
    def _format_context_with_sources(self, retrieved_docs):
        """Format retrieved documents with source information for citations"""
        formatted_context = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source_url = doc.metadata.get('source', 'Unknown source')
            title = doc.metadata.get('title', f'Page {i}')
            content = doc.page_content
            
            # Create formatted context with source info
            context_piece = f"""
Document {i}:
Title: {title}
Source URL: {source_url}
Content: {content}
---"""
            formatted_context.append(context_piece)
        
        return "\n".join(formatted_context)
    
    def _extract_sources_from_response(self, response: str) -> tuple:
        """Extract sources from response and return clean text + sources separately"""
        import re
        
        print(f"🔍 DEBUG - Original response: {response[:200]}...")  # Debug
        
        # Extract citation patterns like [Source: title](url)
        citation_pattern = r'\[Source:\s*([^\]]+)\]\(([^\)]+)\)'
        citations = re.findall(citation_pattern, response)
        print(f"🔍 DEBUG - Found {len(citations)} citation patterns: {citations}")  # Debug
        
        # Also extract numbered citations and URLs
        url_pattern = r'https?://[^\s\)\]]+'
        urls = re.findall(url_pattern, response)
        print(f"🔍 DEBUG - Found {len(urls)} URLs: {urls[:3]}")  # Debug
        
        # Clean the response text (remove all citations)
        clean_response = re.sub(r'\[Source:[^\]]*\]\([^\)]*\)', '', response)
        clean_response = re.sub(r'\[\d+\]', '', clean_response)
        clean_response = re.sub(r'\(https?://[^\)]+\)', '', clean_response)
        clean_response = re.sub(r'https?://[^\s]+', '', clean_response)
        
        # Format sources
        sources = []
        seen_urls = set()
        
        # From citation patterns
        for title, url in citations:
            if url not in seen_urls:
                sources.append({'title': title.strip(), 'url': url.strip()})
                seen_urls.add(url)
        
        # From standalone URLs
        for url in urls:
            if url not in seen_urls:
                sources.append({'title': url, 'url': url})
                seen_urls.add(url)
        
        print(f"🔍 DEBUG - Final sources count: {len(sources)}")  # Debug
        print(f"🔍 DEBUG - Sources: {sources}")  # Debug
        
        return clean_response.strip(), sources
    
    def _create_langgraph_agent(self):
        """
        Create a LangGraph agent with built-in conversation memory
        """
        print("🤖 Creating LangGraph agent with conversation memory...")
        
        def call_rag_agent(state: RAGState):
            """
            Enhanced LangGraph node that handles RAG queries with source extraction
            """
            # Get the latest message
            latest_message = state["messages"][-1]
            
            if isinstance(latest_message, HumanMessage):
                user_input = latest_message.content
                
                # Extract chat history (exclude the current message)
                chat_history = state["messages"][:-1]
                
                # Get relevant documents with source information
                retrieved_docs = self.history_aware_retriever.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # Format context with source information for citations
                formatted_context = self._format_context_with_sources(retrieved_docs)
                
                # Call the RAG chain with enhanced context
                result = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # Extract sources and clean response
                clean_response, sources = self._extract_sources_from_response(result["answer"])
                
                # Create response message with clean text
                response = AIMessage(content=clean_response)
                
                # Update state with sources stored separately
                return {
                    "messages": [response],
                    "context": formatted_context,
                    "last_query": user_input,
                    "sources": sources  # Store sources separately
                }
            else:
                return {"messages": []}
        
        # Build the graph
        builder = StateGraph(RAGState)
        builder.add_node("rag_agent", call_rag_agent)
        builder.add_edge(START, "rag_agent")
        builder.add_edge("rag_agent", END)
        
        # Compile with checkpointer for conversation memory
        self.graph = builder.compile(checkpointer=self.checkpointer)
        
        print("✅ LangGraph agent with conversation memory created!")
    
    def query_with_memory(self, question: str, thread_id: str = "default_conversation"):
        """
        Query the RAG agent with conversation memory
        
        Args:
            question: The question to ask
            thread_id: Conversation thread ID for maintaining separate conversations
            
        Returns:
            Answer with conversation context
        """
        if self.graph is None:
            return "Please load a website first using load_and_process_website()"
        
        print(f"\n🤔 Question: {question}")
        print(f"💬 Thread ID: {thread_id}")
        print("-" * 50)
        
        # Configuration for the conversation thread
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create input message
        input_message = HumanMessage(content=question)
        
        # Invoke the graph with conversation memory
        result = self.graph.invoke(
            {"messages": [input_message]}, 
            config=config
        )
        
        # Get the latest AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            answer = ai_messages[-1].content
            print(f"💡 Answer: {answer}")
            
            # Get full conversation state for accurate count
            final_state = self.graph.get_state(config)
            total_messages = len(final_state.values.get("messages", [])) if final_state.values else 0
            print(f"📚 Conversation history: {total_messages} messages")
            
            return {
                "answer": answer,
                "context": result.get("context", ""),
                "thread_id": thread_id,
                "conversation_length": total_messages
            }
        else:
            return {"answer": "No response generated", "context": "", "thread_id": thread_id}
    
    def query_with_memory_streaming(self, question: str, thread_id: str = "default_conversation") -> Iterator[str]:
        """
        Enhanced streaming query with source extraction (clean text + sources separately)
        
        Args:
            question: The question to ask
            thread_id: Conversation thread ID for maintaining separate conversations
            
        Yields:
            Streaming chunks of the clean response + sources at end
        """
        if self.graph is None:
            yield "Please load a website first using load_and_process_website()"
            return
        
        print(f"\n🤔 Streaming Question: {question}")
        print(f"💬 Thread ID: {thread_id}")
        print("-" * 50)
        
        try:
            # Configuration for the conversation thread
            config = {"configurable": {"thread_id": thread_id}}
            
            # Create input message
            input_message = HumanMessage(content=question)
            
            # Get chat history for context
            current_state = self.graph.get_state(config)
            chat_history = current_state.values.get("messages", []) if current_state.values else []
            
            # Get relevant documents for citations
            retrieved_docs = self.history_aware_retriever.invoke({
                "input": question,
                "chat_history": chat_history
            })
            
            # Use the RAG chain with conversation memory
            rag_result = self.rag_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })
            
            # Extract sources and get clean response
            clean_response, sources = self._extract_sources_from_response(rag_result["answer"])
            
            # FALLBACK: If no sources extracted from response, get them from retrieved docs
            if not sources and retrieved_docs:
                print(f"🚑 DEBUG - No sources in response, extracting from docs...")
                fallback_sources = []
                for i, doc in enumerate(retrieved_docs):
                    source_url = doc.metadata.get('source', '')
                    title = doc.metadata.get('title', f'Document {i+1}')
                    if source_url:
                        fallback_sources.append({'title': title, 'url': source_url})
                
                # Remove duplicates
                seen_urls = set()
                sources = []
                for source in fallback_sources:
                    if source['url'] not in seen_urls:
                        sources.append(source)
                        seen_urls.add(source['url'])
                
                print(f"🚑 DEBUG - Fallback sources: {len(sources)} sources")
            
            # Stream the clean response character by character
            for char in clean_response:
                yield char
                time.sleep(0.01)  # Small delay for visible streaming
            
            # Send sources information at the end (special format)
            if sources:
                yield f"\n\n__SOURCES__:{json.dumps(sources)}"
            
            # Update conversation history with clean response
            ai_message = AIMessage(content=clean_response)
            self.graph.update_state(
                config,
                {"messages": [input_message, ai_message]}
            )
            
            print(f"💡 Streaming response with {len(sources)} sources completed for thread: {thread_id}")
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            yield f"Error generating response: {str(e)}"
    
    def get_conversation_history(self, thread_id: str = "default_conversation"):
        """
        Get the conversation history for a specific thread
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get the current state of the conversation
            state = self.graph.get_state(config)
            messages = state.values.get("messages", []) if state.values else []
            
            print(f"\n📜 Conversation History for Thread: {thread_id}")
            print("=" * 60)
            
            if not messages:
                print("No conversation history found.")
                return []
            
            for i, msg in enumerate(messages, 1):
                if isinstance(msg, HumanMessage):
                    print(f"{i}. 👤 Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"{i}. 🤖 Assistant: {msg.content}")
                print()
            
            print(f"Total messages in conversation: {len(messages)}")
            return messages
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []
    
    def clear_conversation_history(self, thread_id: str = "default_conversation"):
        """
        Clear conversation history for a specific thread
        """
        print(f"🧹 Clearing conversation history for thread: {thread_id}")
        
        try:
            # Configuration for the conversation thread
            config = {"configurable": {"thread_id": thread_id}}
            
            # Update the state with empty messages to clear history
            self.graph.update_state(
                config,
                {"messages": []}
            )
            
            print(f"✅ Conversation history cleared for thread: {thread_id}")
            
        except Exception as e:
            print(f"❌ Error clearing conversation history: {e}")
            print("⚠️ Note: With MemorySaver, restart the application to fully clear memory.")
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"✅ Vector store saved to {path}")
        else:
            print("❌ No vector store to save. Process a website first.")
    
    def load_vector_store(self, path: str):
        """Load a previously saved vector store"""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Recreate the RAG chain and LangGraph agent
            self._create_conversational_rag_chain()
            self._create_langgraph_agent()
            
            print(f"✅ Vector store loaded from {path}")
        except Exception as e:
            raise ValueError(f"Failed to load vector store from {path}: {e}")
    
    def vector_store_exists(self, path: str):
        """Check if a vector store exists at the given path"""
        required_files = ["index.faiss", "index.pkl"]
        return all(os.path.exists(os.path.join(path, file)) for file in required_files)
    
    def get_crawl_statistics(self) -> Dict:
        """Get detailed crawling statistics"""
        stats = self.session_stats.copy()
        stats.update({
            'scraped_pages_count': len(self.scraped_pages),
            'failed_urls_count': len(self.failed_urls),
            'average_content_length': np.mean([len(page.content) for page in self.scraped_pages]) if self.scraped_pages else 0,
            'content_types': list(set([page.content_type for page in self.scraped_pages])),
            'depth_distribution': {
                f"depth_{i}": len([p for p in self.scraped_pages if p.depth == i]) 
                for i in range(max([p.depth for p in self.scraped_pages], default=[-1]) + 1)
            }
        })
        return stats
    
    def get_scraped_urls(self):
        """Get list of URLs that were scraped"""
        return self.scraped_urls

def main():
    """
    Main function with enhanced conversation memory functionality
    """
    try:
        # Initialize RAG agent with memory
        rag_agent = WebScraperRAGAgentWithMemory()
        
        print("=" * 60)
        print("🌐 Welcome to the Web Scraper RAG Agent with Memory!")
        print("   Chat with any website's content - now with conversation history!")
        print("=" * 60)
        
        # Default vector store path
        vector_store_path = "./vector_store"
        
        # Check if vector store already exists
        if rag_agent.vector_store_exists(vector_store_path):
            print(f"\n📁 Found existing vector store at: {vector_store_path}")
            
            choice = input("\nDo you want to:\n1. Load existing vector store\n2. Scrape new website (will overwrite existing)\n\nEnter choice (1/2): ").strip()
            
            if choice == "1":
                print("\n📂 Loading existing vector store...")
                rag_agent.load_vector_store(vector_store_path)
            elif choice == "2":
                print("\n🌐 Scraping new website...")
                url, crawl_depth, max_pages = get_website_details()
                if url:
                    rag_agent.load_and_process_website(url, crawl_depth, max_pages, vector_store_path)
            else:
                print("Invalid choice. Loading existing vector store.")
                rag_agent.load_vector_store(vector_store_path)
        else:
            print(f"\n🌐 No existing vector store found. Please enter a website to scrape.")
            url, crawl_depth, max_pages = get_website_details()
            if url:
                rag_agent.load_and_process_website(url, crawl_depth, max_pages, vector_store_path)
        
        # Show scraped URLs
        scraped_urls = rag_agent.get_scraped_urls()
        if scraped_urls:
            print(f"\n📄 Scraped URLs ({len(scraped_urls)}):")
            for i, url in enumerate(scraped_urls[:5], 1):
                print(f"  {i}. {url}")
            if len(scraped_urls) > 5:
                print(f"  ... and {len(scraped_urls) - 5} more")
        
        # Interactive query loop with conversation memory
        print("\n" + "="*60)
        print("🤖 RAG Agent with Conversation Memory is ready!")
        print("Features:")
        print("  • Remembers conversation context")
        print("  • Multiple conversation threads")
        print("  • History-aware question reformulation")
        print("  • Streaming responses (in web interface)")
        print("\nCommands:")
        print("  • Ask questions naturally")
        print("  • 'history' - View conversation history")
        print("  • 'clear' - Clear conversation history")
        print("  • 'new thread <n>' - Start new conversation thread")
        print("  • 'thread <n>' - Switch to existing thread")
        print("  • 'threads' - List all active threads")
        print("  • 'quit' - Exit")
        print("="*60)
        
        current_thread = "default_conversation"
        active_threads = {current_thread}
        
        while True:
            question = input(f"\n🤖 [{current_thread}] Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'history':
                rag_agent.get_conversation_history(current_thread)
                continue
            elif question.lower() == 'clear':
                rag_agent.clear_conversation_history(current_thread)
                continue
            elif question.lower().startswith('new thread '):
                new_thread = question[11:].strip()
                if new_thread:
                    current_thread = new_thread
                    active_threads.add(current_thread)
                    print(f"🆕 Started new conversation thread: {current_thread}")
                continue
            elif question.lower().startswith('thread '):
                thread_name = question[7:].strip()
                if thread_name:
                    current_thread = thread_name
                    active_threads.add(current_thread)
                    print(f"🔄 Switched to thread: {current_thread}")
                continue
            elif question.lower() == 'threads':
                print(f"\n📋 Active threads: {', '.join(active_threads)}")
                print(f"   Current thread: {current_thread}")
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Query with conversation memory
            result = rag_agent.query_with_memory(question, current_thread)
    
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please check your .env file and ensure all required API keys are set.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

def get_website_details():
    """Helper function to get website details from user"""
    while True:
        url = input("\n🌐 Please enter the website URL: ").strip()
        
        if not url:
            print("Please enter a valid URL.")
            continue
            
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Basic URL validation
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValueError("Invalid URL")
        except:
            print(f"❌ Invalid URL format: {url}")
            retry = input("Would you like to try again? (y/n): ").lower().strip()
            if retry not in ['y', 'yes']:
                return None, 1, 10
            continue
        
        print(f"\n✅ URL: {url}")
        
        # Ask about crawling depth
        print("\n🔍 Crawling options:")
        print("1. Just the main page (fast)")
        print("2. Main page + linked pages (slower, more content)")
        
        depth_choice = input("Choose crawling depth (1/2): ").strip()
        crawl_depth = 2 if depth_choice == "2" else 1
        
        # Ask about max pages
        if crawl_depth == 2:
            max_pages = input("Maximum pages to scrape (default 10): ").strip()
            try:
                max_pages = int(max_pages) if max_pages else 10
                max_pages = min(max_pages, 50)  # Cap at 50 for safety
            except:
                max_pages = 10
        else:
            max_pages = 1
        
        print(f"\n📊 Settings:")
        print(f"   URL: {url}")
        print(f"   Crawl depth: {crawl_depth}")
        print(f"   Max pages: {max_pages}")
        
        confirm = input("\nProceed with these settings? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            return url, crawl_depth, max_pages

if __name__ == "__main__":
    main()

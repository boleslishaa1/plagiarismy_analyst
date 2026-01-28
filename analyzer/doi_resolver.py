"""
DOI resolution and source text retrieval module.
"""

import requests
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import tempfile
import re

from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from .config import SourceDocument, SourceType, API_ENDPOINTS

logger = logging.getLogger(__name__)

class DOIResolver:
    """Resolves DOIs and retrieves source texts"""
    
    def __init__(self, cache_dir: str = "./doi_cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.headers = {
            'User-Agent': 'AcademicPlagiarismAnalyzer/1.0 (mailto:admin@example.com)'
        }
    
    def resolve_doi(self, doi: str) -> Optional[SourceDocument]:
        """Resolve DOI and retrieve source text"""
        doi_clean = doi.strip().lower()
        
        # Check cache first
        if self.use_cache:
            cached = self._get_from_cache(doi_clean)
            if cached:
                logger.info(f"Using cached version for DOI: {doi_clean}")
                return cached
        
        # Try different retrieval methods in order
        source = self._retrieve_via_open_access(doi_clean)
        
        if not source:
            source = self._retrieve_via_api(doi_clean)
        
        if source:
            # Cache the result
            self._save_to_cache(source)
        
        return source
    
    def _get_from_cache(self, doi: str) -> Optional[SourceDocument]:
        """Get document from cache"""
        cache_file = self.cache_dir / f"{hashlib.md5(doi.encode()).hexdigest()}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert source_type string back to enum
                    data['source_type'] = SourceType(data['source_type'])
                    return SourceDocument(**data)
            except Exception as e:
                logger.warning(f"Cache read error for {doi}: {e}")
        
        return None
    
    def _save_to_cache(self, source: SourceDocument):
        """Save document to cache"""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(source.doi.encode()).hexdigest()}.json"
            with open(cache_file, 'w') as f:
                # Convert SourceType enum to string for JSON serialization
                data = source.__dict__.copy()
                data['source_type'] = data['source_type'].value
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _retrieve_via_open_access(self, doi: str) -> Optional[SourceDocument]:
        """Try to retrieve via open access sources first"""
        open_access_sources = [
            f"https://doi.org/{doi}",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{doi}/",
            f"https://link.springer.com/article/{doi}",
            f"https://www.nature.com/articles/{doi.split('/')[-1]}",
        ]
        
        for url in open_access_sources:
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Determine source type and extract text
                    if 'html' in content_type:
                        text, metadata = self._extract_from_html(response.text)
                        source_type = SourceType.HTML
                    elif 'pdf' in content_type:
                        text = self._extract_from_pdf(response.content)
                        metadata = self._extract_metadata_from_html(response.text)
                        source_type = SourceType.PDF
                    elif 'xml' in content_type:
                        text = self._extract_from_xml(response.text)
                        metadata = self._extract_metadata_from_xml(response.text)
                        source_type = SourceType.XML
                    else:
                        text = response.text[:20000]
                        metadata = {}
                        source_type = SourceType.PLAIN_TEXT
                    
                    # Segment sentences
                    sentences = self._segment_text(text)
                    
                    return SourceDocument(
                        doi=doi,
                        title=metadata.get('title', f"Document {doi}"),
                        authors=metadata.get('authors', []),
                        year=metadata.get('year', datetime.now().year),
                        source_type=source_type,
                        full_text=text,
                        sentences=sentences,
                        url=url,
                        retrieval_date=datetime.now().isoformat()
                    )
                    
            except Exception as e:
                logger.debug(f"Failed to retrieve from {url}: {e}")
                continue
        
        return None
    
    def _retrieve_via_api(self, doi: str) -> Optional[SourceDocument]:
        """Retrieve via CrossRef or other APIs"""
        for api_name, api_url in API_ENDPOINTS.items():
            try:
                url = api_url.format(doi=doi)
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse API response
                    if api_name == 'crossref':
                        item = data.get('message', {})
                        title = item.get('title', ['Unknown'])[0] if item.get('title') else 'Unknown'
                        authors = [
                            f"{a.get('given', '')} {a.get('family', '')}".strip()
                            for a in item.get('author', [])
                        ]
                        year = item.get('published-print', {}).get('date-parts', [[0]])[0][0]
                        url = item.get('URL', f"https://doi.org/{doi}")
                    
                    elif api_name == 'openalex':
                        title = data.get('title', 'Unknown')
                        authors = [
                            a.get('author', {}).get('display_name', '')
                            for a in data.get('authorships', [])
                        ]
                        year = data.get('publication_year', datetime.now().year)
                        url = data.get('doi', f"https://doi.org/{doi}")
                    
                    return SourceDocument(
                        doi=doi,
                        title=title,
                        authors=authors,
                        year=year,
                        source_type=SourceType.PLAIN_TEXT,
                        full_text=f"Metadata only for {doi}",
                        sentences=[],
                        url=url,
                        retrieval_date=datetime.now().isoformat()
                    )
                    
            except Exception as e:
                logger.debug(f"API {api_name} failed for {doi}: {e}")
                continue
        
        return None
    
    def _extract_from_html(self, html: str) -> tuple[str, Dict]:
        """Extract text and metadata from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Try to find main content
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|body|article'))
        )
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers
        
        # Extract metadata
        metadata = self._extract_metadata_from_html(html)
        
        return text[:50000], metadata  # Limit text length
    
    def _extract_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
        
        try:
            doc = fitz.open(tmp_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return ' '.join(text_parts)[:50000]
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
        finally:
            import os
            os.unlink(tmp_path)
    
    def _extract_from_xml(self, xml: str) -> str:
        """Extract text from XML"""
        soup = BeautifulSoup(xml, 'xml')
        
        text_parts = []
        for tag in soup.find_all(['p', 'sec', 'abstract', 'body', 'article']):
            if tag.text and len(tag.text.strip()) > 20:
                text_parts.append(tag.text.strip())
        
        return ' '.join(text_parts)[:50000]
    
    def _extract_metadata_from_html(self, html: str) -> Dict:
        """Extract metadata from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {'title': 'Unknown', 'authors': [], 'year': datetime.now().year}
        
        # Try to get title
        title_tag = (
            soup.find('meta', property='og:title') or
            soup.find('meta', {'name': 'citation_title'}) or
            soup.find('title')
        )
        if title_tag:
            metadata['title'] = title_tag.get('content', title_tag.text)
        
        # Try to get authors
        author_tags = soup.find_all('meta', {'name': 'citation_author'})
        if author_tags:
            metadata['authors'] = [tag.get('content', '') for tag in author_tags]
        
        # Try to get year
        year_tag = (
            soup.find('meta', {'name': 'citation_year'}) or
            soup.find('meta', {'name': 'citation_publication_date'}) or
            soup.find('meta', property='article:published_time')
        )
        if year_tag:
            year_str = year_tag.get('content', '')
            if year_str:
                try:
                    metadata['year'] = int(year_str[:4])
                except ValueError:
                    pass
        
        return metadata
    
    def _extract_metadata_from_xml(self, xml: str) -> Dict:
        """Extract metadata from XML"""
        soup = BeautifulSoup(xml, 'xml')
        metadata = {'title': 'Unknown', 'authors': [], 'year': datetime.now().year}
        
        title_tag = soup.find('article-title') or soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.text
        
        author_tags = soup.find_all('contrib')
        if author_tags:
            for author in author_tags:
                name = author.find('surname')
                if name:
                    metadata['authors'].append(name.text)
        
        year_tag = soup.find('year')
        if year_tag:
            try:
                metadata['year'] = int(year_tag.text)
            except ValueError:
                pass
        
        return metadata
    
    def _segment_text(self, text: str) -> List[str]:
        """Segment text into sentences"""
        if not text:
            return []
        
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except:
            # Fallback segmentation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
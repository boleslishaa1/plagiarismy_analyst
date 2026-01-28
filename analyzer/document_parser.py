"""
Document parsing and citation extraction module.
"""

import re
import docx
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from .config import Citation, CitationStyle, Sentence

logger = logging.getLogger(__name__)

class DocumentParser:
    """Parses DOCX documents with citation extraction"""
    
    def __init__(self):
        self.citation_patterns = {
            CitationStyle.APA: r'\(([A-Za-z]+,\s*\d{4})\)',
            CitationStyle.VANCOUVER: r'\[(\d+(?:,\s*\d+)*)\]',
            CitationStyle.IEEE: r'\[(\d+)\]',
            CitationStyle.HARVARD: r'\(([A-Za-z]+\s*et\s*al\.?\s*,\s*\d{4})\)',
        }
        self.doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        
    def parse_docx(self, file_path: str) -> Dict:
        """Parse DOCX file and extract structured content"""
        try:
            doc = docx.Document(file_path)
            
            sections = {}
            current_section = "header"
            current_text = []
            citations_found = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                    
                # Check if paragraph is a section header
                is_section = self._is_section_header(text)
                
                if is_section:
                    if current_text:
                        sections[current_section] = ' '.join(current_text)
                    current_section = text
                    current_text = []
                else:
                    current_text.append(text)
                    
                    # Extract citations
                    citations = self._extract_citations(text)
                    citations_found.extend(citations)
            
            if current_text:
                sections[current_section] = ' '.join(current_text)
                
            # Extract metadata
            metadata = {
                'author': doc.core_properties.author or "Unknown",
                'created': doc.core_properties.created,
                'modified': doc.core_properties.modified,
                'word_count': sum(len(s.split()) for s in sections.values())
            }
            
            return {
                'sections': sections,
                'citations': citations_found,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header"""
        lower_text = text.lower()
        
        # Check for common section headers
        for header in SECTION_HEADERS:
            if header in lower_text:
                return True
        
        # Check formatting clues
        is_all_caps = text.isupper()
        is_short = len(text.split()) < 5
        has_no_punctuation = text[-1] not in '.!?'
        
        return (is_all_caps and is_short) or (is_short and has_no_punctuation)
    
    def _extract_citations(self, text: str) -> List[Citation]:
        """Extract citations from text"""
        citations = []
        
        for style, pattern in self.citation_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = Citation(
                    id=f"cit_{len(citations) + 1}",
                    text=match.group(0),
                    position=(match.start(), match.end())
                )
                citations.append(citation)
        
        return citations
    
    def segment_sentences(self, text: str, section: str = "") -> List[Sentence]:
        """Scientific-grade sentence segmentation"""
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback simple segmentation
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segmented = []
        for idx, sent in enumerate(sentences):
            # Skip very short sentences
            if len(sent.strip()) < 10:
                continue
                
            word_count = len(word_tokenize(sent))
            
            segmented.append(Sentence(
                id=f"{section}_{idx}" if section else f"sent_{idx}",
                text=sent.strip(),
                section=section,
                position=idx,
                word_count=word_count,
                citations=[]
            ))
            
        return segmented
    
    def extract_dois_from_references(self, references_text: str) -> List[str]:
        """Extract DOIs from references section"""
        if not references_text:
            return []
        
        # Find DOIs in the text
        dois = re.findall(self.doi_pattern, references_text, re.IGNORECASE)
        
        # Clean and deduplicate
        cleaned_dois = []
        seen = set()
        for doi in dois:
            doi_lower = doi.lower().strip()
            if doi_lower not in seen:
                seen.add(doi_lower)
                cleaned_dois.append(doi_lower)
        
        return cleaned_dois

# Export the SECTION_HEADERS for use in other modules
from .config import SECTION_HEADERS
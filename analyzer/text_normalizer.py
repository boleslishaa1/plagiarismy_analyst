"""
Text normalization and preprocessing module.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Normalizes text for comparison"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        try:
            self.stemmer = nltk.PorterStemmer()
        except:
            nltk.download('punkt')
            self.stemmer = nltk.PorterStemmer()
        
        # Academic stop words (common in papers)
        self.academic_stop_words = {
            'however', 'therefore', 'moreover', 'furthermore',
            'consequently', 'nevertheless', 'nonetheless',
            'respectively', 'namely', 'i.e.', 'e.g.', 'cf.',
            'et al', 'ibid', 'op cit', 'loc cit'
        }
        
        # Regex patterns for normalization
        self.patterns = [
            (r'\d+', 'NUM'),  # Replace numbers
            (r'[^\w\s]', ''),  # Remove punctuation
            (r'\s+', ' '),  # Normalize whitespace
        ]
        
        # Common abbreviations in academic text
        self.abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'cf.': 'compare',
            'et al.': 'and others',
            'etc.': 'and so on',
            'viz.': 'namely',
        }
    
    def normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Apply regex patterns
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        
        # Tokenize and process
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stop words and stem
        processed_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in self.academic_stop_words and
                len(token) > 1):
                try:
                    stemmed = self.stemmer.stem(token)
                    processed_tokens.append(stemmed)
                except:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def get_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Generate n-grams from text"""
        tokens = self._tokenize(text)
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            return text.split()
    
    def remove_citations(self, text: str) -> str:
        """Remove citation markers from text"""
        # Remove citation patterns
        citation_patterns = [
            r'\([A-Za-z]+,\s*\d{4}\)',  # APA
            r'\[\d+(?:,\s*\d+)*\]',     # Vancouver
            r'\[\d+\]',                  # IEEE
            r'\([A-Za-z]+\s*et\s*al\.?\s*,\s*\d{4}\)',  # Harvard
        ]
        
        for pattern in citation_patterns:
            text = re.sub(pattern, '', text)
        
        return text.strip()
    
    def calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts"""
        words1 = set(self._tokenize(self.normalize(text1)))
        words2 = set(self._tokenize(self.normalize(text2)))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
"""
Configuration, enums, and data classes for the plagiarism analyzer.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import json

class CitationStyle(Enum):
    """Supported citation styles"""
    APA = "apa"
    VANCOUVER = "vancouver"
    IEEE = "ieee"
    HARVARD = "harvard"
    UNKNOWN = "unknown"

class PlagiarismType(Enum):
    """Plagiarism classification types"""
    EXACT = "exact_plagiarism"
    NEAR_VERBATIM = "near_verbatim"
    PARAPHRASE = "strong_paraphrasing"
    ACCEPTABLE = "acceptable"
    UNCITED_SIMILARITY = "uncited_similarity"

class SourceType(Enum):
    """Source document types"""
    HTML = "html"
    XML = "xml"
    PDF = "pdf"
    PLAIN_TEXT = "plain_text"

@dataclass
class Citation:
    """Citation reference in text"""
    id: str
    text: str
    doi: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    position: Tuple[int, int] = (0, 0)

@dataclass
class Sentence:
    """Manuscript sentence with metadata"""
    id: str
    text: str
    section: str
    position: int
    word_count: int
    citations: List[Citation]
    normalized_text: str = ""
    embeddings: Optional[List[float]] = None

@dataclass
class SourceDocument:
    """Retrieved source document"""
    doi: str
    title: str
    authors: List[str]
    year: int
    source_type: SourceType
    full_text: str
    sentences: List[str]
    url: str
    retrieval_date: str

@dataclass
class SimilarityResult:
    """Sentence similarity result"""
    sentence_id: str
    source_doi: str
    source_sentence: str
    lexical_score: float
    semantic_score: float
    combined_score: float
    plagiarism_type: PlagiarismType
    matched_words: List[str]
    source_url: str

@dataclass
class SectionAnalysis:
    """Analysis results for a section"""
    name: str
    total_words: int
    plagiarized_words: int
    sentences: List[Sentence]
    similarity_results: List[SimilarityResult]
    section_similarity: float = 0.0

@dataclass
class AnalysisReport:
    """Complete analysis report"""
    document_hash: str
    analysis_date: str
    total_words: int
    total_sentences: int
    sections: List[SectionAnalysis]
    overall_similarity: float
    flagged_sentences: int
    plagiarism_breakdown: Dict[PlagiarismType, int]
    processing_time: float
    references_checked: int

    def to_json(self) -> str:
        """Convert report to JSON string"""
        return json.dumps(asdict(self), indent=2, default=str)
    
    def save_json(self, filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

# Constants
THRESHOLDS = {
    'exact_plagiarism': 0.85,
    'near_verbatim': 0.70,
    'strong_paraphrasing': 0.55,
    'acceptable': 0.0
}

COLOR_MAP = {
    PlagiarismType.EXACT: (1, 0, 0),        # Red
    PlagiarismType.NEAR_VERBATIM: (1, 0.65, 0),  # Orange
    PlagiarismType.PARAPHRASE: (1, 1, 0),   # Yellow
    PlagiarismType.UNCITED_SIMILARITY: (0.5, 0, 0.5),  # Purple
    PlagiarismType.ACCEPTABLE: (0, 0, 0)    # Black
}

API_ENDPOINTS = {
    'crossref': 'https://api.crossref.org/works/{doi}',
    'datacite': 'https://api.datacite.org/works/{doi}',
    'openalex': 'https://api.openalex.org/works/https://doi.org/{doi}',
}

SECTION_HEADERS = [
    'abstract', 'introduction', 'method', 'methodology',
    'results', 'discussion', 'conclusion', 'references'
]
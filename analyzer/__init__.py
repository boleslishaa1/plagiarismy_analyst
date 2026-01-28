"""
Academic Plagiarism Analyzer Package
"""

from .plagiarism_analyzer import AcademicPlagiarismAnalyzer
from .config import (
    CitationStyle, PlagiarismType, SourceType,
    Citation, Sentence, SourceDocument,
    SimilarityResult, SectionAnalysis, AnalysisReport
)
from .document_parser import DocumentParser
from .doi_resolver import DOIResolver
from .text_normalizer import TextNormalizer
from .similarity_analyzer import SimilarityAnalyzer
from .report_generator import ReportGenerator

__version__ = "1.0.0"
__author__ = "Academic Integrity Systems"
__license__ = "MIT"

__all__ = [
    'AcademicPlagiarismAnalyzer',
    'CitationStyle', 'PlagiarismType', 'SourceType',
    'Citation', 'Sentence', 'SourceDocument',
    'SimilarityResult', 'SectionAnalysis', 'AnalysisReport',
    'DocumentParser', 'DOIResolver', 'TextNormalizer',
    'SimilarityAnalyzer', 'ReportGenerator'
]
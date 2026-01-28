#!/usr/bin/env python3
"""
Main entry point for the Academic Plagiarism Analyzer.
Can be used as a module or directly from command line.
"""

import sys

if __name__ == "__main__":
    # If called directly, use the CLI
    from cli import main
    main()
else:
    # If imported as a module, expose the main classes
    from analyzer import (
        AcademicPlagiarismAnalyzer,
        DocumentParser,
        DOIResolver,
        SimilarityAnalyzer,
        ReportGenerator
    )
    
    __all__ = [
        'AcademicPlagiarismAnalyzer',
        'DocumentParser',
        'DOIResolver',
        'SimilarityAnalyzer',
        'ReportGenerator'
    ]
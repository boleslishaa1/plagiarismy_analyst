"""
Validation functions for the plagiarism analyzer.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

def validate_doi(doi: str) -> Tuple[bool, Optional[str]]:
    """
    Validate DOI format.
    Returns (is_valid, cleaned_doi)
    """
    if not doi:
        return False, None
    
    # Basic DOI pattern
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    
    # Clean the DOI
    cleaned = doi.strip().lower()
    
    # Remove 'doi:' prefix if present
    if cleaned.startswith('doi:'):
        cleaned = cleaned[4:].strip()
    
    # Remove URL prefix if present
    if cleaned.startswith('http'):
        parsed = urlparse(cleaned)
        if parsed.path:
            cleaned = parsed.path.lstrip('/')
    
    # Validate format
    if re.match(doi_pattern, cleaned, re.IGNORECASE):
        return True, cleaned
    else:
        return False, None

def validate_file(filepath: str, allowed_extensions: list = None) -> Tuple[bool, Optional[str]]:
    """
    Validate file exists and has correct extension.
    Returns (is_valid, error_message)
    """
    if not filepath:
        return False, "File path is empty"
    
    path = Path(filepath)
    
    if not path.exists():
        return False, f"File does not exist: {filepath}"
    
    if not path.is_file():
        return False, f"Path is not a file: {filepath}"
    
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            return False, f"File must have one of these extensions: {allowed_extensions}"
    
    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if path.stat().st_size > max_size:
        return False, f"File is too large (max {max_size/1024/1024:.0f}MB)"
    
    return True, None

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_citation(citation_text: str, style: str) -> bool:
    """Validate citation format based on style"""
    patterns = {
        'apa': r'^\([A-Z][a-z]+,\s*\d{4}\)$',
        'vancouver': r'^\[\d+(?:,\s*\d+)*\]$',
        'ieee': r'^\[\d+\]$',
        'harvard': r'^\([A-Z][a-z]+\s+et\s+al\.?,\s*\d{4}\)$',
    }
    
    if style.lower() not in patterns:
        return False
    
    return bool(re.match(patterns[style.lower()], citation_text))
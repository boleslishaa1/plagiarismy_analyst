"""
Helper functions and utilities.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import logging
from datetime import datetime

def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"plagiarism_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def calculate_hash(filepath: str) -> str:
    """Calculate MD5 hash of a file"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = text.replace('"', "'")
    
    return text.strip()

def format_percentage(value: float) -> str:
    """Format float as percentage"""
    return f"{value:.2%}"

def format_time(seconds: float) -> str:
    """Format seconds as human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_file_size(filepath: str) -> str:
    """Get human-readable file size"""
    size = Path(filepath).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} TB"

def create_directory_structure(base_dir: str) -> Dict[str, Path]:
    """Create standard directory structure"""
    base = Path(base_dir)
    
    directories = {
        'reports': base / 'reports',
        'cache': base / 'cache',
        'logs': base / 'logs',
        'temp': base / 'temp',
        'output': base / 'output',
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories
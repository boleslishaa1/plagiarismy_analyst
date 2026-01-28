"""
Command-line interface for the plagiarism analyzer.
"""

import argparse
import sys
from pathlib import Path
import logging

from analyzer import AcademicPlagiarismAnalyzer
from utils.helpers import setup_logging

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Academic Plagiarism Analyzer - Reference-Aware Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze research_paper.docx
  %(prog)s analyze paper.docx --output ./my_reports --no-cache
  %(prog)s batch ./documents --workers 4
  %(prog)s test --all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single document')
    analyze_parser.add_argument('document', help='Path to DOCX document')
    analyze_parser.add_argument('--output', '-o', default='./reports',
                               help='Output directory for reports')
    analyze_parser.add_argument('--cache', '-c', default='./cache',
                               help='Cache directory for DOI resolution')
    analyze_parser.add_argument('--no-cache', action='store_true',
                               help='Disable caching')
    analyze_parser.add_argument('--prefix', '-p', default='',
                               help='Prefix for output filenames')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple documents')
    batch_parser.add_argument('directory', help='Directory containing DOCX files')
    batch_parser.add_argument('--output', '-o', default='./batch_reports',
                             help='Output directory')
    batch_parser.add_argument('--cache', '-c', default='./cache',
                             help='Cache directory')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--all', action='store_true',
                            help='Run all tests')
    test_parser.add_argument('--unit', action='store_true',
                            help='Run unit tests only')
    test_parser.add_argument('--integration', action='store_true',
                            help='Run integration tests only')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configure system settings')
    config_parser.add_argument('--clear-cache', action='store_true',
                              help='Clear DOI cache')
    config_parser.add_argument('--list-cache', action='store_true',
                              help='List cached DOIs')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    if args.command == 'analyze':
        run_analysis(args, logger)
    elif args.command == 'batch':
        run_batch(args, logger)
    elif args.command == 'test':
        run_tests(args, logger)
    elif args.command == 'config':
        run_config(args, logger)
    elif args.command == 'version':
        show_version()

def run_analysis(args, logger):
    """Run single document analysis"""
    from utils.validators import validate_file
    
    # Validate input file
    is_valid, error = validate_file(args.document, ['.docx', '.doc'])
    if not is_valid:
        print(f"Error: {error}")
        sys.exit(1)
    
    print(f"\n📄 Analyzing document: {Path(args.document).name}")
    print("="*50)
    
    try:
        analyzer = AcademicPlagiarismAnalyzer(
            cache_dir=args.cache,
            use_cache=not args.no_cache,
            output_dir=args.output
        )
        
        report = analyzer.analyze(args.document, args.prefix)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)

def run_batch(args, logger):
    """Run batch analysis"""
    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    docx_files = list(dir_path.glob("*.docx"))
    if not docx_files:
        print(f"Error: No DOCX files found in {args.directory}")
        sys.exit(1)
    
    print(f"\n📁 Batch analyzing {len(docx_files)} documents from {args.directory}")
    print("="*50)
    
    try:
        analyzer = AcademicPlagiarismAnalyzer(
            cache_dir=args.cache,
            use_cache=True,
            output_dir=args.output
        )
        
        results = analyzer.batch_analyze(args.directory)
        
        print(f"\n✅ Batch analysis completed!")
        print(f"   Processed: {len(results)} documents")
        print(f"   Reports saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        print(f"\n❌ Batch analysis failed: {e}")
        sys.exit(1)

def run_tests(args, logger):
    """Run tests"""
    print("\n🧪 Running tests...")
    print("="*50)
    
    import pytest
    import os
    
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    pytest_args = []
    
    if args.unit:
        pytest_args.append(str(test_dir / "test_document_parser.py"))
        pytest_args.append(str(test_dir / "test_similarity_analyzer.py"))
    elif args.integration:
        pytest_args.append(str(test_dir / "test_integration.py"))
    elif args.all:
        pytest_args.append(str(test_dir))
    else:
        pytest_args.append(str(test_dir))
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)

def run_config(args, logger):
    """Run configuration commands"""
    if args.clear_cache:
        import shutil
        cache_dir = Path("./cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared")
        else:
            print("ℹ️  Cache directory does not exist")
    
    if args.list_cache:
        cache_dir = Path("./cache")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"\n📚 Cached DOIs ({len(cache_files)} files):")
            for file in cache_files[:10]:  # Show first 10
                print(f"  • {file.stem}")
            if len(cache_files) > 10:
                print(f"  ... and {len(cache_files) - 10} more")
        else:
            print("ℹ️  Cache directory does not exist")

def show_version():
    """Show version information"""
    from analyzer import __version__, __author__, __license__
    
    print(f"\n📊 Academic Plagiarism Analyzer")
    print("="*30)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nA reference-aware plagiarism detection system")
    print("comparable in rigor to commercial solutions.")

if __name__ == "__main__":
    main()
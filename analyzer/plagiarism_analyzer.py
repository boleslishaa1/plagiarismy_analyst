"""
Main analyzer class coordinating the plagiarism detection pipeline.
"""

import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

from .document_parser import DocumentParser
from .doi_resolver import DOIResolver
from .similarity_analyzer import SimilarityAnalyzer
from .report_generator import ReportGenerator
from .config import (
    AnalysisReport, SectionAnalysis, PlagiarismType,
    Sentence, Citation
)

logger = logging.getLogger(__name__)

class AcademicPlagiarismAnalyzer:
    """Main analyzer class coordinating all components"""
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 use_cache: bool = True,
                 output_dir: str = "./reports"):
        
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.use_cache = use_cache
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.parser = DocumentParser()
        self.doi_resolver = DOIResolver(str(self.cache_dir), use_cache)
        self.analyzer = SimilarityAnalyzer()
        self.report_generator = ReportGenerator(str(self.output_dir))
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'dois_resolved': 0,
            'sentences_analyzed': 0,
            'api_calls': 0
        }
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "plagiarism_analysis.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def analyze(self, docx_path: str, report_prefix: str = "") -> AnalysisReport:
        """Main analysis pipeline"""
        start_time = datetime.now()
        
        logger.info(f"Starting analysis of {docx_path}")
        
        # Generate document hash
        with open(docx_path, 'rb') as f:
            document_hash = hashlib.md5(f.read()).hexdigest()
        
        # 1. Parse document
        logger.info("Step 1: Parsing document...")
        doc_data = self.parser.parse_docx(docx_path)
        
        # 2. Extract and segment sentences
        logger.info("Step 2: Segmenting sentences...")
        sections_analysis = []
        all_sentences = []
        
        for section_name, section_text in doc_data['sections'].items():
            sentences = self.parser.segment_sentences(section_text, section_name)
            
            # Map citations to sentences
            for sentence in sentences:
                sentence.citations = self._map_citations_to_sentence(
                    sentence.text, 
                    doc_data['citations']
                )
                all_sentences.append(sentence)
            
            sections_analysis.append(SectionAnalysis(
                name=section_name,
                total_words=sum(s.word_count for s in sentences),
                plagiarized_words=0,
                sentences=sentences,
                similarity_results=[],
                section_similarity=0.0
            ))
        
        self.stats['sentences_analyzed'] = len(all_sentences)
        
        # 3. Extract DOIs from references section
        logger.info("Step 3: Extracting DOIs...")
        dois = []
        for section in sections_analysis:
            if 'reference' in section.name.lower():
                # Combine sentences to get references text
                ref_text = ' '.join([s.text for s in section.sentences])
                dois.extend(self.parser.extract_dois_from_references(ref_text))
        
        # 4. Resolve DOIs and retrieve sources
        logger.info(f"Step 4: Resolving {len(dois)} DOIs...")
        source_docs = []
        for doi in dois[:10]:  # Limit to first 10 DOIs for performance
            try:
                source = self.doi_resolver.resolve_doi(doi)
                if source:
                    source_docs.append(source)
                    self.stats['dois_resolved'] += 1
                    logger.info(f"  Resolved: {doi}")
            except Exception as e:
                logger.warning(f"Failed to resolve DOI {doi}: {e}")
        
        # 5. Analyze similarity
        logger.info("Step 5: Analyzing similarity...")
        total_plagiarized_words = 0
        plagiarism_breakdown = {ptype: 0 for ptype in PlagiarismType}
        
        # Prepare source data for batch processing
        source_data = [(doc.doi, doc.sentences) for doc in source_docs]
        
        for section in sections_analysis:
            if not section.sentences or not source_data:
                continue
            
            # Compute similarity for this section
            results = self.analyzer.compute_similarity_batch(
                section.sentences,
                source_data
            )
            
            # Update section with results
            section.similarity_results = results
            
            # Update statistics
            for result in results:
                if result.plagiarism_type != PlagiarismType.ACCEPTABLE:
                    # Find the sentence and update word counts
                    for sent in section.sentences:
                        if sent.id == result.sentence_id:
                            section.plagiarized_words += sent.word_count
                            total_plagiarized_words += sent.word_count
                            break
                    
                    plagiarism_breakdown[result.plagiarism_type] += 1
            
            # Calculate section similarity
            if section.total_words > 0:
                section.section_similarity = section.plagiarized_words / section.total_words
        
        # 6. Prepare final report
        logger.info("Step 6: Preparing final report...")
        total_words = doc_data['metadata']['word_count']
        overall_similarity = total_plagiarized_words / total_words if total_words > 0 else 0
        flagged_sentences = sum(plagiarism_breakdown.values())
        processing_time = (datetime.now() - start_time).total_seconds()
        
        report = AnalysisReport(
            document_hash=document_hash,
            analysis_date=datetime.now().isoformat(),
            total_words=total_words,
            total_sentences=len(all_sentences),
            sections=sections_analysis,
            overall_similarity=overall_similarity,
            flagged_sentences=flagged_sentences,
            plagiarism_breakdown=plagiarism_breakdown,
            processing_time=processing_time,
            references_checked=len(source_docs)
        )
        
        # 7. Generate outputs
        logger.info("Step 7: Generating reports...")
        if not report_prefix:
            report_prefix = f"analysis_{document_hash[:8]}"
        
        output_files = self.report_generator.generate_all_reports(report, report_prefix)
        
        # Log completion
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        logger.info(f"Overall similarity: {overall_similarity:.2%}")
        logger.info(f"Flagged sentences: {flagged_sentences}")
        
        # Print summary to console
        self._print_summary(report)
        
        return report
    
    def _map_citations_to_sentence(self, 
                                  sentence_text: str, 
                                  citations: List[Citation]) -> List[Citation]:
        """Map citations to the sentence they appear in"""
        matched_citations = []
        for citation in citations:
            if citation.text in sentence_text:
                matched_citations.append(citation)
        return matched_citations
    
    def _print_summary(self, report: AnalysisReport):
        """Print analysis summary to console"""
        print("\n" + "="*60)
        print("ACADEMIC PLAGIARISM ANALYSIS - SUMMARY")
        print("="*60)
        print(f"Document Hash: {report.document_hash[:16]}")
        print(f"Analysis Date: {report.analysis_date}")
        print(f"Total Words: {report.total_words:,}")
        print(f"Total Sentences: {report.total_sentences}")
        print(f"References Checked: {report.references_checked}")
        print(f"Processing Time: {report.processing_time:.2f} seconds")
        print("\n" + "-"*60)
        print(f"OVERALL SIMILARITY: {report.overall_similarity:.2%}")
        print(f"FLAGGED SENTENCES: {report.flagged_sentences}")
        print("-"*60)
        
        if report.flagged_sentences > 0:
            print("\nPLAGIARISM BREAKDOWN:")
            for ptype, count in report.plagiarism_breakdown.items():
                if count > 0 and ptype != PlagiarismType.ACCEPTABLE:
                    pct = (count / report.flagged_sentences * 100) if report.flagged_sentences > 0 else 0
                    print(f"  {ptype.value.replace('_', ' ').title():20} {count:3d} ({pct:5.1f}%)")
        
        print("\nSECTION ANALYSIS:")
        for section in report.sections:
            if section.total_words > 50:  # Only show substantial sections
                print(f"  {section.name[:30]:30} {section.section_similarity:6.2%} "
                      f"({section.plagiarized_words:,}/{section.total_words:,} words)")
        
        print("\n" + "="*60)
        print("Reports saved to:", self.output_dir)
        print("="*60)
    
    def batch_analyze(self, documents_dir: str) -> Dict[str, AnalysisReport]:
        """Analyze multiple documents in batch"""
        documents_dir = Path(documents_dir)
        if not documents_dir.exists():
            raise ValueError(f"Directory not found: {documents_dir}")
        
        docx_files = list(documents_dir.glob("*.docx"))
        if not docx_files:
            raise ValueError(f"No DOCX files found in {documents_dir}")
        
        results = {}
        
        for docx_file in docx_files:
            try:
                logger.info(f"Processing: {docx_file.name}")
                report = self.analyze(str(docx_file), f"batch_{docx_file.stem}")
                results[docx_file.name] = report
                
                # Save individual report
                summary_file = self.output_dir / f"batch_summary_{docx_file.stem}.json"
                with open(summary_file, 'w') as f:
                    json.dump(report.__dict__, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Failed to analyze {docx_file.name}: {e}")
                print(f"✗ Error analyzing {docx_file.name}: {e}")
        
        # Generate batch summary
        self._generate_batch_summary(results)
        
        return results
    
    def _generate_batch_summary(self, results: Dict[str, AnalysisReport]):
        """Generate summary report for batch analysis"""
        summary_data = []
        
        for filename, report in results.items():
            summary_data.append({
                'filename': filename,
                'document_hash': report.document_hash[:16],
                'total_words': report.total_words,
                'overall_similarity': report.overall_similarity,
                'flagged_sentences': report.flagged_sentences,
                'references_checked': report.references_checked,
                'processing_time': report.processing_time
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "batch_summary.csv"
            df.to_csv(summary_file, index=False)
            
            # Also save as JSON
            json_file = self.output_dir / "batch_summary.json"
            with open(json_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Batch summary saved: {summary_file}")
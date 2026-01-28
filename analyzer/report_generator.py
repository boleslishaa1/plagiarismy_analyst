"""
Report generation module for PDF, CSV, and visualization outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .config import (
    AnalysisReport, PlagiarismType, SectionAnalysis,
    SimilarityResult, COLOR_MAP
)

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates annotated PDF and reports"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize reportlab fonts
        try:
            pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
            self.font_name = 'DejaVu'
        except:
            self.font_name = 'Helvetica'
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#2E4053')
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))
        
        # Highlighted text style
        self.styles.add(ParagraphStyle(
            name='Highlighted',
            parent=self.styles['Normal'],
            fontSize=10,
            backColor=colors.yellow,
            spaceAfter=6
        ))
    
    def generate_all_reports(self, report: AnalysisReport, prefix: str = ""):
        """Generate all report formats"""
        if not prefix:
            prefix = f"report_{report.document_hash[:8]}"
        
        # Generate PDF report
        pdf_path = self.output_dir / f"{prefix}.pdf"
        self.generate_pdf_report(report, str(pdf_path))
        
        # Generate CSV report
        csv_path = self.output_dir / f"{prefix}_detailed.csv"
        self.generate_csv_report(report, str(csv_path))
        
        # Generate visualization
        viz_path = self.output_dir / f"{prefix}_visualization.png"
        self.generate_visualization(report, str(viz_path))
        
        # Generate JSON summary
        json_path = self.output_dir / f"{prefix}_summary.json"
        report.save_json(str(json_path))
        
        logger.info(f"Reports generated at: {self.output_dir}")
        return {
            'pdf': str(pdf_path),
            'csv': str(csv_path),
            'png': str(viz_path),
            'json': str(json_path)
        }
    
    def generate_pdf_report(self, report: AnalysisReport, output_path: str):
        """Generate annotated PDF report"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Title page
        story.append(Paragraph("ACADEMIC PLAGIARISM ANALYSIS REPORT", 
                              self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Metadata table
        metadata = [
            ["Analysis Date", report.analysis_date],
            ["Document Hash", report.document_hash[:16]],
            ["Total Words", f"{report.total_words:,}"],
            ["Total Sentences", str(report.total_sentences)],
            ["Overall Similarity", f"{report.overall_similarity:.2%}"],
            ["Flagged Sentences", str(report.flagged_sentences)],
            ["References Checked", str(report.references_checked)],
            ["Processing Time", f"{report.processing_time:.2f} seconds"],
        ]
        
        metadata_table = Table(metadata, colWidths=[200, 200])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Plagiarism Breakdown
        story.append(Paragraph("PLAGIARISM BREAKDOWN", self.styles['CustomHeading']))
        
        breakdown_data = [["Type", "Sentences", "Percentage"]]
        total_flagged = sum(report.plagiarism_breakdown.values())
        
        color_mapping = {
            PlagiarismType.EXACT: colors.red,
            PlagiarismType.NEAR_VERBATIM: colors.orange,
            PlagiarismType.PARAPHRASE: colors.yellow,
            PlagiarismType.UNCITED_SIMILARITY: colors.purple,
        }
        
        row = 1
        for ptype, count in report.plagiarism_breakdown.items():
            if count > 0 and ptype != PlagiarismType.ACCEPTABLE:
                pct = (count / total_flagged * 100) if total_flagged > 0 else 0
                breakdown_data.append([
                    ptype.value.replace('_', ' ').title(),
                    str(count),
                    f"{pct:.1f}%"
                ])
                row += 1
        
        breakdown_table = Table(breakdown_data, colWidths=[200, 100, 100])
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
        ]
        
        # Add color coding
        for i, ptype in enumerate(report.plagiarism_breakdown.keys()):
            if i < len(breakdown_data) - 1:
                table_style.append(('BACKGROUND', (0, i+1), (-1, i+1), 
                                   color_mapping.get(ptype, colors.white)))
        
        breakdown_table.setStyle(TableStyle(table_style))
        story.append(breakdown_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Section Analysis
        story.append(Paragraph("SECTION ANALYSIS", self.styles['CustomHeading']))
        
        section_data = [["Section", "Words", "Plagiarized", "Similarity"]]
        for section in report.sections:
            if section.total_words > 0:
                section_data.append([
                    section.name[:30] + "..." if len(section.name) > 30 else section.name,
                    f"{section.total_words:,}",
                    f"{section.plagiarized_words:,}",
                    f"{section.section_similarity:.2%}"
                ])
        
        section_table = Table(section_data, colWidths=[250, 80, 80, 80])
        section_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
        ]))
        story.append(section_table)
        
        # Detailed findings (new page)
        story.append(PageBreak())
        story.append(Paragraph("DETAILED FINDINGS", self.styles['CustomHeading']))
        
        # Add sentence-level findings
        detailed_data = [["Sentence ID", "Similarity", "Type", "Source DOI"]]
        for section in report.sections:
            for result in section.similarity_results:
                if result.plagiarism_type != PlagiarismType.ACCEPTABLE:
                    detailed_data.append([
                        result.sentence_id,
                        f"{result.combined_score:.3f}",
                        result.plagiarism_type.value.replace('_', ' ').title(),
                        result.source_doi[:30] + "..." if len(result.source_doi) > 30 else result.source_doi
                    ])
        
        if len(detailed_data) > 1:
            detailed_table = Table(detailed_data, colWidths=[100, 80, 100, 200])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ]))
            story.append(detailed_table)
        else:
            story.append(Paragraph("No significant similarities found.", 
                                  self.styles['BodyText']))
        
        # Add visualization if exists
        viz_path = self.output_dir / f"report_{report.document_hash[:8]}_visualization.png"
        if viz_path.exists():
            story.append(PageBreak())
            story.append(Paragraph("VISUALIZATION", self.styles['CustomHeading']))
            story.append(Image(str(viz_path), width=6*inch, height=4.5*inch))
        
        # Legend
        story.append(PageBreak())
        story.append(Paragraph("LEGEND", self.styles['CustomHeading']))
        
        legend_data = [
            ["Color", "Plagiarism Type", "Threshold", "Description"],
            ["█", "Exact Plagiarism", "≥ 0.85", "Verbatim copy without quotation"],
            ["█", "Near-Verbatim", "0.70 - 0.85", "Minor modifications to original"],
            ["█", "Strong Paraphrasing", "0.55 - 0.70", "Substantial rephrasing"],
            ["█", "Uncited Similarity", "≥ 0.55", "High similarity without citation"],
        ]
        
        legend_table = Table(legend_data, colWidths=[30, 150, 80, 200])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
        ]))
        story.append(legend_table)
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
    
    def generate_csv_report(self, report: AnalysisReport, output_path: str):
        """Generate CSV report for detailed analysis"""
        rows = []
        
        for section in report.sections:
            for result in section.similarity_results:
                if result.plagiarism_type != PlagiarismType.ACCEPTABLE:
                    # Find the original sentence
                    original_sentence = ""
                    for sent in section.sentences:
                        if sent.id == result.sentence_id:
                            original_sentence = sent.text
                            break
                    
                    rows.append({
                        'sentence_id': result.sentence_id,
                        'section': section.name,
                        'original_sentence': original_sentence[:500],
                        'source_doi': result.source_doi,
                        'similarity_score': result.combined_score,
                        'plagiarism_type': result.plagiarism_type.value,
                        'lexical_score': result.lexical_score,
                        'semantic_score': result.semantic_score,
                        'matched_words_count': len(result.matched_words),
                        'matched_words_sample': ', '.join(result.matched_words[:10]),
                        'source_sentence': result.source_sentence[:500],
                        'source_url': result.source_url
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"CSV report generated: {output_path}")
        else:
            logger.info("No significant similarities found for CSV report")
    
    def generate_visualization(self, report: AnalysisReport, output_path: str):
        """Generate visualization of results"""
        plt.figure(figsize=(15, 10))
        
        # 1. Similarity distribution
        plt.subplot(2, 2, 1)
        all_scores = []
        for section in report.sections:
            for result in section.similarity_results:
                if result.plagiarism_type != PlagiarismType.ACCEPTABLE:
                    all_scores.append(result.combined_score)
        
        if all_scores:
            plt.hist(all_scores, bins=20, edgecolor='black', alpha=0.7, 
                    color='skyblue')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Similarity Score Distribution')
            plt.axvline(x=0.55, color='red', linestyle='--', 
                       label='Threshold (0.55)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Plagiarism type breakdown
        plt.subplot(2, 2, 2)
        types = []
        counts = []
        colors_list = []
        
        for ptype, count in report.plagiarism_breakdown.items():
            if count > 0 and ptype != PlagiarismType.ACCEPTABLE:
                types.append(ptype.value.replace('_', ' ').title())
                counts.append(count)
                # Get color from COLOR_MAP
                rgb = COLOR_MAP.get(ptype, (0, 0, 0))
                colors_list.append(rgb)
        
        if counts:
            plt.pie(counts, labels=types, colors=colors_list,
                   autopct='%1.1f%%', startangle=90)
            plt.title('Plagiarism Type Breakdown')
        
        # 3. Section similarity
        plt.subplot(2, 2, 3)
        sections = []
        similarities = []
        
        for section in report.sections:
            if section.total_words > 50:  # Only show substantial sections
                sections.append(section.name[:15] + '...' if len(section.name) > 15 else section.name)
                similarities.append(section.section_similarity)
        
        if sections:
            # Create colormap from green to red
            cmap = LinearSegmentedColormap.from_list(
                'similarity', ['green', 'yellow', 'red']
            )
            bar_colors = [cmap(min(sim * 2, 1)) for sim in similarities]  # Scale for better color variation
            
            bars = plt.bar(range(len(sections)), similarities, color=bar_colors)
            plt.xlabel('Section')
            plt.ylabel('Similarity Index')
            plt.title('Section Similarity Index')
            plt.xticks(range(len(sections)), sections, rotation=45, ha='right')
            plt.axhline(y=0.55, color='red', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar, sim in zip(bars, similarities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{sim:.1%}', ha='center', va='bottom', fontsize=8)
            
            plt.grid(True, alpha=0.3, axis='y')
        
        # 4. Timeline of matches
        plt.subplot(2, 2, 4)
        if all_scores:
            sorted_scores = sorted(all_scores, reverse=True)
            plt.plot(range(len(sorted_scores)), sorted_scores, 
                    marker='o', markersize=3, linestyle='-', 
                    alpha=0.7, linewidth=1)
            plt.xlabel('Sentence Rank (by similarity)')
            plt.ylabel('Similarity Score')
            plt.title('Similarity Scores (Sorted)')
            plt.axhline(y=0.55, color='red', linestyle='--', 
                       label='Threshold')
            plt.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5)
            plt.axhline(y=0.85, color='darkred', linestyle='--', alpha=0.5)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Plagiarism Analysis Report\n'
                    f'Overall Similarity: {report.overall_similarity:.2%} | '
                    f'Flagged Sentences: {report.flagged_sentences}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization generated: {output_path}")
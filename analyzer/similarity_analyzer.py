"""
Similarity computation and plagiarism classification module.
"""

import numpy as np
from typing import List, Tuple
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Sentence, SimilarityResult, PlagiarismType, THRESHOLDS
from .text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """Computes similarity between sentences"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 lexical_weight: float = 0.4,
                 semantic_weight: float = 0.6):
        
        self.normalizer = TextNormalizer()
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        
        # Initialize embedding model
        self.model = self._load_embedding_model(model_name)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            stop_words='english',
            analyzer='word'
        )
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    def _load_embedding_model(self, model_name: str):
        """Load sentence embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            return SentenceTransformer(model_name)
        except ImportError as e:
            logger.warning(f"Cannot load SentenceTransformer: {e}")
            logger.info("Using TF-IDF only (no semantic embeddings)")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def compute_similarity(self, 
                          manuscript_sentences: List[Sentence],
                          source_sentences: List[str],
                          source_id: str) -> List[SimilarityResult]:
        """Compute similarity between manuscript and source sentences"""
        if not source_sentences:
            return []
        
        # Normalize all sentences
        manuscript_texts = [s.text for s in manuscript_sentences]
        manuscript_norm = [self.normalizer.normalize(t) for t in manuscript_texts]
        source_norm = [self.normalizer.normalize(s) for s in source_sentences]
        
        # Filter out empty sentences
        valid_indices = []
        valid_manuscript = []
        for i, (orig, norm) in enumerate(zip(manuscript_texts, manuscript_norm)):
            if len(norm.split()) >= 3:  # Minimum 3 words after normalization
                valid_indices.append(i)
                valid_manuscript.append(orig)
        
        if not valid_manuscript:
            return []
        
        # Compute lexical similarity
        lexical_scores = self._compute_lexical_similarity(
            valid_manuscript, source_sentences
        )
        
        # Compute semantic similarity
        semantic_scores = self._compute_semantic_similarity(
            valid_manuscript, source_sentences
        )
        
        # Combine scores and create results
        results = []
        for idx, ms_sentence in enumerate(manuscript_sentences):
            if idx not in valid_indices:
                # Skip very short sentences
                results.append(SimilarityResult(
                    sentence_id=ms_sentence.id,
                    source_doi=source_id,
                    source_sentence="",
                    lexical_score=0.0,
                    semantic_score=0.0,
                    combined_score=0.0,
                    plagiarism_type=PlagiarismType.ACCEPTABLE,
                    matched_words=[],
                    source_url=f"https://doi.org/{source_id}"
                ))
                continue
            
            # Get scores for this sentence
            rel_idx = valid_indices.index(idx)
            lexical_score = lexical_scores[rel_idx]
            semantic_score = semantic_scores[rel_idx]
            
            # Combine scores
            combined = (lexical_score * self.lexical_weight + 
                       semantic_score * self.semantic_weight)
            
            # Find best matching source sentence
            best_match_idx = np.argmax(lexical_scores[rel_idx])
            
            # Get matched words
            matched_words = self._get_matched_words(
                manuscript_norm[idx],
                source_norm[best_match_idx]
            )
            
            # Classify plagiarism
            plagiarism_type = self._classify_plagiarism(combined)
            
            results.append(SimilarityResult(
                sentence_id=ms_sentence.id,
                source_doi=source_id,
                source_sentence=source_sentences[best_match_idx],
                lexical_score=float(lexical_score),
                semantic_score=float(semantic_score),
                combined_score=float(combined),
                plagiarism_type=plagiarism_type,
                matched_words=matched_words,
                source_url=f"https://doi.org/{source_id}"
            ))
        
        return results
    
    def _compute_lexical_similarity(self, 
                                   text1_list: List[str], 
                                   text2_list: List[str]) -> np.ndarray:
        """Compute lexical similarity using TF-IDF"""
        # Combine all texts for TF-IDF
        all_texts = text1_list + text2_list
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Split matrix
        tfidf1 = tfidf_matrix[:len(text1_list)]
        tfidf2 = tfidf_matrix[len(text1_list):]
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf1, tfidf2)
        
        # Get maximum similarity for each text1 against all text2
        max_similarities = np.max(similarity_matrix, axis=1)
        
        return max_similarities
    
    def _compute_semantic_similarity(self,
                                    text1_list: List[str],
                                    text2_list: List[str]) -> np.ndarray:
        """Compute semantic similarity using embeddings"""
        if not self.model:
            # Return lexical scores if no model
            return self._compute_lexical_similarity(text1_list, text2_list)
        
        try:
            # Generate embeddings
            embeddings1 = self.model.encode(text1_list, show_progress_bar=False)
            embeddings2 = self.model.encode(text2_list, show_progress_bar=False)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            # Get maximum similarity for each text1 against all text2
            max_similarities = np.max(similarity_matrix, axis=1)
            
            return max_similarities
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            # Fallback to lexical similarity
            return self._compute_lexical_similarity(text1_list, text2_list)
    
    def _get_matched_words(self, text1: str, text2: str) -> List[str]:
        """Find matching words between two normalized texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        return list(words1.intersection(words2))
    
    def _classify_plagiarism(self, score: float) -> PlagiarismType:
        """Classify similarity score into plagiarism type"""
        if score >= THRESHOLDS['exact_plagiarism']:
            return PlagiarismType.EXACT
        elif score >= THRESHOLDS['near_verbatim']:
            return PlagiarismType.NEAR_VERBATIM
        elif score >= THRESHOLDS['strong_paraphrasing']:
            return PlagiarismType.PARAPHRASE
        else:
            return PlagiarismType.ACCEPTABLE
    
    def compute_similarity_batch(self,
                                manuscript_sentences: List[Sentence],
                                source_documents: List[Tuple[str, List[str]]]) -> List[SimilarityResult]:
        """Compute similarity against multiple source documents"""
        all_results = []
        
        for source_id, source_sentences in source_documents:
            results = self.compute_similarity(
                manuscript_sentences,
                source_sentences[:200],  # Limit to first 200 sentences
                source_id
            )
            all_results.extend(results)
        
        # For each manuscript sentence, keep only the highest similarity result
        best_results = {}
        for result in all_results:
            sentence_id = result.sentence_id
            if (sentence_id not in best_results or 
                result.combined_score > best_results[sentence_id].combined_score):
                best_results[sentence_id] = result
        
        return list(best_results.values())
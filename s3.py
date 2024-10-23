import numpy as np
from sklearn.linear_model import LinearRegression
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

class S3ScoreExplained:
    """
    S3 (Summary Scoring System) combines multiple metrics to evaluate summary quality:
    1. Content Coverage (ROUGE scores)
    2. Semantic Similarity (using embeddings)
    3. Summary Quality Features
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_content_coverage(self, reference, summary):
        """
        Measure how well the summary covers the content of original text
        using ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation)
        """
        scores = self.rouge_scorer.score(reference, summary)
        
        # Get different ROUGE metrics
        rouge1 = scores['rouge1'].fmeasure  # Unigram overlap
        rouge2 = scores['rouge2'].fmeasure  # Bigram overlap
        rougeL = scores['rougeL'].fmeasure  # Longest common subsequence
        
        # Combine ROUGE scores
        content_score = np.mean([rouge1, rouge2, rougeL])
        
        return {
            'content_score': content_score,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL
        }

    def calculate_semantic_similarity(self, reference_embedding, summary_embedding):
        """
        Measure semantic similarity between reference and summary
        using cosine similarity of their embeddings
        """
        similarity = cosine_similarity(
            reference_embedding.reshape(1, -1),
            summary_embedding.reshape(1, -1)
        )[0][0]
        
        return similarity

    def calculate_quality_features(self, reference, summary):
        """
        Calculate summary quality features:
        1. Length ratio (summary shouldn't be too long or too short)
        2. Density (information packed per word)
        3. Coverage (what fraction of important parts are included)
        """
        # Length ratio scoring
        ideal_ratio = 0.3  # Summary should be ~30% of original
        actual_ratio = len(summary.split()) / len(reference.split())
        length_score = 1.0 - abs(ideal_ratio - actual_ratio)
        
        # Density scoring (unique information per word)
        unique_words_summary = len(set(summary.lower().split()))
        total_words_summary = len(summary.split())
        density_score = unique_words_summary / total_words_summary
        
        # Coverage scoring (important terms from reference in summary)
        ref_important_words = set(w.lower() for w in reference.split() if len(w) > 4)
        sum_important_words = set(w.lower() for w in summary.split() if len(w) > 4)
        coverage = len(ref_important_words.intersection(sum_important_words)) / len(ref_important_words)
        
        return {
            'length_score': length_score,
            'density_score': density_score,
            'coverage_score': coverage
        }

    def calculate_s3_score(self, reference, summary, reference_embedding, summary_embedding):
        """
        Calculate final S3 score by combining all components:
        - Content coverage (ROUGE scores)
        - Semantic similarity
        - Quality features
        
        Returns both final score and detailed breakdown
        """
        # Get individual components
        content_scores = self.calculate_content_coverage(reference, summary)
        semantic_score = self.calculate_semantic_similarity(reference_embedding, summary_embedding)
        quality_scores = self.calculate_quality_features(reference, summary)
        
        # Combine scores using weighted average
        weights = {
            'content': 0.4,     # Content coverage importance
            'semantic': 0.3,    # Semantic similarity importance
            'quality': 0.3      # Summary quality importance
        }
        
        final_score = (
            weights['content'] * content_scores['content_score'] +
            weights['semantic'] * semantic_score +
            weights['quality'] * np.mean([
                quality_scores['length_score'],
                quality_scores['density_score'],
                quality_scores['coverage_score']
            ])
        )
        
        # Return detailed breakdown
        return {
            'final_s3_score': final_score,
            'content_scores': content_scores,
            'semantic_score': semantic_score,
            'quality_scores': quality_scores,
            'component_weights': weights
        }

# Example usage:
def evaluate_summary(reference, summary, embeddings_model):
    """Example of how to use the S3 score system"""
    scorer = S3ScoreExplained()
    
    # Get embeddings (pseudo-code - replace with actual embedding method)
    ref_embedding = embeddings_model.encode(reference)
    sum_embedding = embeddings_model.encode(summary)
    
    # Calculate S3 score
    results = scorer.calculate_s3_score(reference, summary, ref_embedding, sum_embedding)
    
    return results
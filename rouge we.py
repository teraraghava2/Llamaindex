import torch
from metrics.rouge_score import compute_rouge

def compute_rouge_we(transcript_chunks, summary_chunks, bert_model):
    """
    Compute ROUGE-WE score using BERT embeddings
    """
    def get_embeddings(chunks, bert_model):
        max_len = min(512, max(len(chunk) for chunk in chunks))
        padded_chunks = [
            chunk[:max_len] + [0] * (max_len - len(chunk)) 
            for chunk in chunks
        ]
        attention_masks = [
            [1] * len(chunk[:max_len]) + [0] * (max_len - len(chunk[:max_len]))
            for chunk in chunks
        ]
        
        inputs = {
            'input_ids': torch.tensor(padded_chunks),
            'attention_mask': torch.tensor(attention_masks)
        }
        
        # Get token-level embeddings (not just CLS)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use last hidden states for token embeddings
            token_embeddings = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]
            
        return token_embeddings, attention_masks

    # Get embeddings for both transcript and summary
    transcript_embeddings, transcript_masks = get_embeddings(transcript_chunks, bert_model)
    summary_embeddings, summary_masks = get_embeddings(summary_chunks, bert_model)
    
    # Calculate word embedding similarity matrix
    def compute_we_similarity(ref_embeddings, hyp_embeddings, ref_mask, hyp_mask):
        # Normalize embeddings
        ref_norm = torch.nn.functional.normalize(ref_embeddings, p=2, dim=-1)
        hyp_norm = torch.nn.functional.normalize(hyp_embeddings, p=2, dim=-1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(ref_norm, hyp_norm.transpose(-1, -2))
        
        # Apply attention masks
        mask_matrix = torch.matmul(
            ref_mask.float().unsqueeze(-1),
            hyp_mask.float().unsqueeze(-2)
        )
        similarity_matrix = similarity_matrix * mask_matrix
        
        return similarity_matrix

    # Calculate ROUGE-WE scores
    def calculate_rouge_we_scores(similarity_matrix, ref_mask, hyp_mask):
        # Precision: For each hypothesis word, take max similarity with reference words
        precision_scores = similarity_matrix.max(dim=1)[0]
        precision_scores = precision_scores * hyp_mask.float()
        precision = precision_scores.sum() / (hyp_mask.float().sum() + 1e-12)
        
        # Recall: For each reference word, take max similarity with hypothesis words
        recall_scores = similarity_matrix.max(dim=2)[0]
        recall_scores = recall_scores * ref_mask.float()
        recall = recall_scores.sum() / (ref_mask.float().sum() + 1e-12)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

    total_rouge_we = 0
    total_chunks = 0
    
    # Calculate ROUGE-WE for each chunk pair
    for i in range(len(transcript_embeddings)):
        similarity_matrix = compute_we_similarity(
            transcript_embeddings[i:i+1],
            summary_embeddings[i:i+1],
            transcript_masks[i:i+1],
            summary_masks[i:i+1]
        )
        
        scores = calculate_rouge_we_scores(
            similarity_matrix[0],
            transcript_masks[i],
            summary_masks[i]
        )
        
        total_rouge_we += scores['f1']
        total_chunks += 1
    
    rouge_we_score = total_rouge_we / total_chunks if total_chunks > 0 else 0
    return rouge_we_score

def compute_s3_score(transcript_chunks, summary_chunks, bert_model):
    """
    Compute S3 score using ROUGE-WE and semantic similarity
    """
    # 1. Calculate ROUGE-WE score
    rouge_we_score = compute_rouge_we(transcript_chunks, summary_chunks, bert_model)
    
    # 2. Calculate semantic similarity
    with torch.no_grad():
        # Prepare inputs for semantic similarity
        max_len = min(512, max(len(chunk) for chunk in transcript_chunks + summary_chunks))
        
        def prepare_bert_inputs(chunks):
            padded_chunks = [
                chunk[:max_len] + [0] * (max_len - len(chunk)) 
                for chunk in chunks
            ]
            attention_masks = [
                [1] * len(chunk[:max_len]) + [0] * (max_len - len(chunk[:max_len]))
                for chunk in chunks
            ]
            return {
                'input_ids': torch.tensor(padded_chunks),
                'attention_mask': torch.tensor(attention_masks)
            }
        
        # Get CLS embeddings for semantic similarity
        transcript_inputs = prepare_bert_inputs(transcript_chunks)
        summary_inputs = prepare_bert_inputs(summary_chunks)
        
        transcript_embedding = bert_model(**transcript_inputs)[0][:, 0, :].mean(dim=0)
        summary_embedding = bert_model(**summary_inputs)[0][:, 0, :].mean(dim=0)
        
        semantic_similarity = torch.nn.functional.cosine_similarity(
            transcript_embedding.unsqueeze(0),
            summary_embedding.unsqueeze(0)
        ).item()
    
    # 3. Combine scores (equal weights)
    s3_score = 0.5 * rouge_we_score + 0.5 * semantic_similarity
    
    return {
        'rouge_we': rouge_we_score,
        'semantic_similarity': semantic_similarity,
        's3_score': s3_score
    }
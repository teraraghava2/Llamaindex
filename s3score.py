from metrics.rouge_score import compute_rouge
import torch

def compute_s3_score(transcript_chunks, summary_chunks, bert_model=None, tokenizer=None):
    """
    Compute S3 score using pre-computed BERT model, tokenizer, and chunks
    
    Args:
        transcript_chunks: Pre-tokenized transcript chunks from generate_reports
        summary_chunks: Pre-tokenized summary chunks from generate_reports
        bert_model: Optional pre-loaded BERT model instance
        tokenizer: Optional pre-loaded BERT tokenizer instance
    """
    # Reuse existing rouge computation
    rouge_scores = compute_rouge(transcript_chunks, summary_chunks)
    
    # If model and tokenizer aren't passed, we'll get them from the parent context
    if bert_model is None or tokenizer is None:
        from transformers import BertModel, BertTokenizer
        model_name = "bert-base-uncased"
        bert_model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Since chunks are already tokenized, we can directly get embeddings
    with torch.no_grad():
        transcript_embeddings = bert_model(**tokenizer(transcript_chunks, 
                                                     return_tensors="pt", 
                                                     padding=True, 
                                                     truncation=True))[0][:, 0, :]
        
        summary_embeddings = bert_model(**tokenizer(summary_chunks,
                                                  return_tensors="pt",
                                                  padding=True,
                                                  truncation=True))[0][:, 0, :]
    
    # Calculate semantic similarity using cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(transcript_embeddings.mean(dim=0),
                                                  summary_embeddings.mean(dim=0),
                                                  dim=0)
    
    # Combine ROUGE and semantic scores (you can adjust weights as needed)
    s3_score = 0.5 * (rouge_scores['rouge1']['f'] + cos_sim.item())
    
    return s3_score
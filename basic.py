import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from rouge_score import rouge_scorer
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mover_score import get_idf_dict, word_mover_score
from scipy.spatial.distance import cosine

nltk.download('punkt')

# Initialize tokenizers and models for various metrics
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute ROUGE scores
def compute_rouge(transcript, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(transcript, summary)
    return scores

# Function to compute BLEU scores
def compute_bleu(transcript, summary):
    bleu = sacrebleu.corpus_bleu([summary], [[transcript]])
    return bleu.score

# Function to compute BertScore
def compute_bert_score(transcript, summary):
    P, R, F1 = bert_score([summary], [transcript], model_type="bert-base-uncased", lang="en", rescale_with_baseline=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# Function to compute METEOR scores
def compute_meteor(transcript, summary):
    meteor = meteor_score([transcript], summary)
    return meteor

# Function to compute CHRF
def compute_chrf(transcript, summary):
    chrf = sacrebleu.corpus_chrf([summary], [[transcript]])
    return chrf.score

# Function to compute Sentence Mover's Similarity (SMS)
def sentence_mover_similarity(transcript, summary):
    doc_tokens = nltk.sent_tokenize(transcript)
    sum_tokens = nltk.sent_tokenize(summary)
    
    doc_embeds = bert_model(**tokenizer(doc_tokens, return_tensors="pt", padding=True, truncation=True))['last_hidden_state'].mean(dim=1)
    sum_embeds = bert_model(**tokenizer(sum_tokens, return_tensors="pt", padding=True, truncation=True))['last_hidden_state'].mean(dim=1)
    
    sim_scores = cosine_similarity(sum_embeds.detach().numpy(), doc_embeds.detach().numpy()).mean()
    return sim_scores

# Function to compute MoverScore
def compute_mover_score(transcript, summary):
    idf_dict_hyp = get_idf_dict([summary])
    idf_dict_ref = get_idf_dict([transcript])
    
    score = word_mover_score([transcript], [summary], idf_dict_ref, idf_dict_hyp, tokenizer)
    return np.mean(score)

# Function to compute BLANC (simplified version from before)
def blanc_score(document, summary):
    inputs = tokenizer(document, return_tensors='pt')
    mask_indices = torch.rand(inputs['input_ids'].shape).argsort(dim=1)[:, :5]  # Randomly mask 5 tokens
    inputs['input_ids'].scatter_(1, mask_indices, tokenizer.mask_token_id)
    
    with torch.no_grad():
        outputs_no_summary = bert_model(**inputs)
    
    logits_no_summary = outputs_no_summary.last_hidden_state.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, bert_model.config.hidden_size))
    
    inputs_with_summary = tokenizer(document + ' ' + summary, return_tensors='pt')
    inputs_with_summary['input_ids'].scatter_(1, mask_indices, tokenizer.mask_token_id)
    
    with torch.no_grad():
        outputs_with_summary = bert_model(**inputs_with_summary)
    
    logits_with_summary = outputs_with_summary.last_hidden_state.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, bert_model.config.hidden_size))
    
    score_diff = (logits_with_summary - logits_no_summary).mean().item()
    return score_diff

# Function to compute SUPERT (simplified)
def supert_score(document, summary):
    doc_tokens = nltk.sent_tokenize(document)
    sum_tokens = nltk.sent_tokenize(summary)
    
    doc_embeds = bert_model(**tokenizer(doc_tokens, return_tensors="pt", padding=True, truncation=True))['last_hidden_state'].mean(dim=1)
    sum_embeds = bert_model(**tokenizer(sum_tokens, return_tensors="pt", padding=True, truncation=True))['last_hidden_state'].mean(dim=1)
    
    sim_scores = cosine_similarity(sum_embeds.detach().numpy(), doc_embeds.detach().numpy()).mean()
    return sim_scores

# Function to compute S3 (placeholder logic)
def compute_s3(transcript, summary):
    # Assume a custom method for calculating S3 here
    # Placeholder logic, adjust according to your implementation
    return 0.5  # Static score for now

# Function to generate a JSON report
def generate_report(transcript, summary):
    report = {}
    report['ROUGE'] = compute_rouge(transcript, summary)
    report['BLEU'] = compute_bleu(transcript, summary)
    report['BertScore'] = compute_bert_score(transcript, summary)
    report['METEOR'] = compute_meteor(transcript, summary)
    report['CHRF'] = compute_chrf(transcript, summary)
    report['MoverScore'] = compute_mover_score(transcript, summary)
    report['SMS'] = sentence_mover_similarity(transcript, summary)
    report['S3'] = compute_s3(transcript, summary)
    report['BLANC'] = blanc_score(transcript, summary)
    report['SUPERT'] = supert_score(transcript, summary)
    
    return report

# Function to read from Excel and process records
def process_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Loop through each record in the dataframe
    reports = []
    for index, row in df.iterrows():
        transcript = row['call transcript']
        summary = row['summarizer']
        
        # Generate the evaluation report
        report = generate_report(transcript, summary)
        
        # Add the report to the list
        reports.append({
            'record_index': index,
            'call_transcript': transcript,
            'summary': summary,
            'evaluation_report': report
        })

    # Output all reports to a JSON file
    with open('evaluation_report.json', 'w') as f:
        json.dump(reports, f, indent=4)
    
    print("Evaluation report saved to 'evaluation_report.json'.")

# Example usage
if __name__ == "__main__":
    # Provide the path to your Excel file here
    excel_file_path = 'path_to_your_file.xlsx'
    process_excel(excel_file_path)

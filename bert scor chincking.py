import pandas as pd
from phoenix.util.model_util import getModel
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

# Load the model and tokenizer once
model_path = getModel("Complaints", "bert-base-uncased")  # Ensure this function returns the correct model path
bert_tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path)

# Initialize the BERTScorer once
bs = BERTScorer(model_type=model_path, num_layers=12, batch_size=256, device=3, idf=False, idf_sents=None)

# Function to split text into chunks with a sliding window
def sliding_window_tokenizer(text, tokenizer, max_len=512, stride=256):
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    total_length = len(tokenized_text)
    
    chunks = []
    for i in range(0, total_length, stride):
        # Append chunks that are within the max_len limit
        chunk = tokenized_text[i:i + max_len]
        if len(chunk) == 0:
            break
        chunks.append(chunk)
    
    return chunks

# Function to compute BERTScore with sliding window for long inputs
def compute_bert_score_with_sliding_window(transcript, summary, bs, tokenizer, max_len=512, stride=256):
    # Tokenize and split into sliding window chunks
    transcript_chunks = sliding_window_tokenizer(transcript, tokenizer, max_len, stride)
    summary_chunks = sliding_window_tokenizer(summary, tokenizer, max_len, stride)

    # Initialize accumulators for precision, recall, and F1
    P_total, R_total, F1_total = 0, 0, 0
    count = 0

    # Loop through chunks and compute BERTScore for each pair of chunks
    for t_chunk, s_chunk in zip(transcript_chunks, summary_chunks):
        # Decode token IDs back to text
        t_text = tokenizer.decode(t_chunk, skip_special_tokens=True)
        s_text = tokenizer.decode(s_chunk, skip_special_tokens=True)
        
        # Compute BERTScore for the current chunk
        P, R, F1 = bs.score([s_text], [t_text], model_type=model_path, lang="en", rescale_with_baseline=True)
        
        # Accumulate scores
        P_total += P.mean().item()
        R_total += R.mean().item()
        F1_total += F1.mean().item()
        count += 1

    # Return the average of all chunks
    return {
        "precision": P_total / count,
        "recall": R_total / count,
        "f1": F1_total / count
    }

# Read the Excel file
filepath = "Summarizer_Annotation.xlsx"
df = pd.read_excel(filepath)
df = df.head()  # Display first few rows, adjust as needed

# Iterate through the DataFrame and compute BERTScore for each row with sliding window
for index, row in df.iterrows():
    transcript = row['call transcript']
    summary = row['summarizer']
    
    # Compute BERTScore using sliding window
    bert_score_result = compute_bert_score_with_sliding_window(transcript, summary, bs, bert_tokenizer)
    
    print(f"Row {index}: {bert_score_result}")
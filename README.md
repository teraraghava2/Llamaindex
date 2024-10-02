# Llamaindex
Llamaindex
import nltk
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine

# Download the necessary nltk resources
nltk.download('punkt')

# Initialize tokenizer and model for sentence embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get sentence embeddings using BERT
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over token embeddings to get sentence embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to extract salient sentences from the source document (pseudo-references)
def extract_salient_sentences(source, n=3):
    sentences = nltk.sent_tokenize(source)
    # Sort sentences by length as a simple heuristic (you can implement more sophisticated methods)
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    return sorted_sentences[:n]  # Return top-n longest sentences as pseudo-references

# Function to compute the SUPERT score
def supert_score(source, summary):
    # Extract pseudo-reference sentences from the source document
    pseudo_references = extract_salient_sentences(source)
    
    # Get sentence embeddings for pseudo-references and the summary
    pseudo_embeddings = [get_sentence_embedding(sent) for sent in pseudo_references]
    summary_embedding = get_sentence_embedding(summary)
    
    # Calculate cosine similarity between the summary and each pseudo-reference
    similarities = [1 - cosine(summary_embedding, ref_emb) for ref_emb in pseudo_embeddings]
    
    # SUPERT score is the mean similarity across all pseudo-references
    return np.mean(similarities)

# Example usage:
source_document = """The quick brown fox jumps over the lazy dog. The dog was surprised but remained calm. 
The fox quickly moved away from the scene after jumping. The dog barked once but stayed in its spot."""
generated_summary = "A fox jumps over a dog and quickly moves away."

supert = supert_score(source_document, generated_summary)
print(f"SUPERT score: {supert}")

from sklearn.linear_model import LinearRegression
import numpy as np
from rouge_score import rouge_scorer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate ROUGE scores
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return np.mean([scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure])

# Function to calculate ROUGE-WE using word embeddings
def calculate_rouge_we(reference, summary, model):
    reference_vector = np.mean([model.wv[word] for word in reference.split() if word in model.wv], axis=0)
    summary_vector = np.mean([model.wv[word] for word in summary.split() if word in model.wv], axis=0)
    return cosine_similarity([reference_vector], [summary_vector])[0][0]

# Train a Word2Vec model for ROUGE-WE (this is an example, you can load pre-trained models)
sentences = [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
word2vec_model = Word2Vec(sentences, vector_size=100, min_count=1)

# Function to calculate S3 score
def s3_score(reference, summary, model, regression_model):
    # Extract features
    rouge_score = calculate_rouge(reference, summary)
    rouge_we_score = calculate_rouge_we(reference, summary, model)
    
    # Combine features (you can add more features like JS-divergence)
    features = np.array([rouge_score, rouge_we_score]).reshape(1, -1)
    
    # Predict using regression model (previously trained on TAC data)
    s3 = regression_model.predict(features)
    return s3

# Train the regression model on human-annotated data (dummy example here)
# In a real scenario, you'd train this on real human judgment datasets
X_train = np.array([[0.5, 0.6], [0.7, 0.8], [0.3, 0.4]])  # Features (ROUGE and ROUGE-WE)
y_train = np.array([0.55, 0.75, 0.35])  # Human scores
regression_model = LinearRegression().fit(X_train, y_train)

S3

# Example usage:
reference = "The quick brown fox jumps over the lazy dog."
summary = "A fox jumps over a lazy dog."
s3 = s3_score(reference, summary, word2vec_model, regression_model)
print(f"S3 score: {s3[0]}")

from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cdist
import numpy as np
import torch

# Initialize tokenizer and model for sentence embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embeddings(text):
    # Tokenize and get the sentence embeddings
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over token embeddings to get sentence embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def sentence_movers_similarity(reference, summary):
    # Get sentence embeddings
    ref_embeddings = get_sentence_embeddings(reference)
    sum_embeddings = get_sentence_embeddings(summary)
    
    # Compute Earth Mover's Distance between sentence embeddings
    distance_matrix = cdist([ref_embeddings], [sum_embeddings], metric='euclidean')
    
    # Return the minimum distance as a similarity score (can be normalized as needed)
    return distance_matrix[0][0]

# Example usage:
reference = "The quick brown fox jumps over the lazy dog."
summary = "A fox jumps over a lazy dog."
sms_score = sentence_movers_similarity(reference, summary)
print(f"SMS score: {sms_score}")

from transformers import BertTokenizer, BertForMaskedLM
import torch

# Initialize the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def blanc_score(document, summary):
    # Mask a portion of the document (random masking can be implemented for robustness)
    inputs = tokenizer(document, return_tensors='pt')
    mask_indices = torch.rand(inputs['input_ids'].shape).argsort(dim=1)[:, :5]  # Randomly mask 5 tokens
    inputs['input_ids'].scatter_(1, mask_indices, tokenizer.mask_token_id)
    
    # Model prediction without summary
    with torch.no_grad():
        outputs_no_summary = model(**inputs)
    
    # Masked tokens probability without summary
    logits_no_summary = outputs_no_summary.logits.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, model.config.vocab_size))
    
    # Integrate summary by concatenating it to the document
    inputs_with_summary = tokenizer(document + ' ' + summary, return_tensors='pt')
    inputs_with_summary['input_ids'].scatter_(1, mask_indices, tokenizer.mask_token_id)
    
    # Model prediction with summary
    with torch.no_grad():
        outputs_with_summary = model(**inputs_with_summary)
    
    # Masked tokens probability with summary
    logits_with_summary = outputs_with_summary.logits.gather(1, mask_indices.unsqueeze(-1).expand(-1, -1, model.config.vocab_size))
    
    # Compare masked token prediction confidence (simplified comparison)
    score_diff = (logits_with_summary - logits_no_summary).mean().item()
    
    return score_diff

# Example usage:
document = "The quick brown fox jumps over the lazy dog."
summary = "A fox jumps over a dog."
blanc = blanc_score(document, summary)
print(f"BLANC score: {blanc}")


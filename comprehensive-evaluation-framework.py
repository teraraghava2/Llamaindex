import pandas as pd
import numpy as np
import openai
from typing import List, Dict
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Excel file containing call transcripts, summaries, and human annotations."""
    return pd.read_excel(file_path)

def generate_evaluation(transcript: str, summary: str) -> Dict[str, int]:
    """Generate evaluation scores using GPT model."""
    prompt = f"""
    Given the following call transcript and summary, evaluate on a scale of 1-5 (1 being lowest, 5 being highest) for each of the following criteria:

    1. Transcript Quality
    2. Summary Quality
    3. Summary Coherence
    4. Resolution Capture
    5. Informative Content
    6. Truthfulness
    7. Absence of Hallucinations

    Also, provide a binary evaluation (0 for No, 1 for Yes) for:
    8. Presence of Toxic Language

    Call Transcript:
    {transcript}

    Summary:
    {summary}

    Provide your evaluation as a Python dictionary with the criteria as keys and scores as values.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in evaluating call transcripts and summaries."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the evaluation dictionary from the response
    evaluation = eval(response.choices[0].message.content)
    return evaluation

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe and add AI-generated evaluation scores."""
    evaluations = []

    for _, row in df.iterrows():
        evaluation = generate_evaluation(row['call_transcript'], row['summary'])
        evaluations.append(evaluation)

    # Convert the list of dictionaries to a DataFrame
    eval_df = pd.DataFrame(evaluations)

    # Rename columns to match the desired label names
    eval_df.columns = [
        'ai_transcript_quality',
        'ai_summary_quality',
        'ai_summary_coherence',
        'ai_resolution_capture',
        'ai_informative_content',
        'ai_truthfulness',
        'ai_absence_of_hallucinations',
        'ai_toxic_language_presence'
    ]

    # Concatenate the original DataFrame with the evaluation DataFrame
    return pd.concat([df, eval_df], axis=1)

def one_hot_encode(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """Perform one-hot encoding on specified categorical columns."""
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df[categorical_columns])
    
    # Create new column names for encoded features
    new_columns = [f"{col}_{val}" for col, vals in zip(categorical_columns, encoder.categories_) for val in vals]
    
    # Create a new dataframe with encoded features
    encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=df.index)
    
    # Concatenate the original dataframe with the encoded features
    return pd.concat([df, encoded_df], axis=1)

def calculate_correlations(df: pd.DataFrame, ai_columns: List[str], human_columns: List[str]) -> pd.DataFrame:
    """Calculate Spearman rank correlations between AI and human annotations."""
    correlations = []

    for ai_col in ai_columns:
        for human_col in human_columns:
            spearman, _ = spearmanr(df[ai_col], df[human_col])
            correlations.append({
                'AI Column': ai_col,
                'Human Column': human_col,
                'Spearman Correlation': spearman
            })

    return pd.DataFrame(correlations)

def create_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """Create and save a heatmap of correlations between AI and human annotations."""
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap: AI Evaluations vs Human Annotations')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load the data
    file_path = "path_to_your_excel_file.xlsx"
    df = load_data(file_path)

    # Process the data and get AI-generated evaluation scores
    result_df = process_data(df)

    # Define categorical columns for one-hot encoding
    categorical_columns = [
        'evaluate the agent notes quality',
        'did the agent notes contain foul or toxic language',
        'did the summary contain foul or toxic language',
        'did you think the resolution was captured in the summary'
    ]

    # Perform one-hot encoding
    encoded_df = one_hot_encode(result_df, categorical_columns)

    # Define the columns for correlation analysis
    ai_columns = [
        'ai_transcript_quality',
        'ai_summary_quality',
        'ai_summary_coherence',
        'ai_resolution_capture',
        'ai_informative_content',
        'ai_truthfulness',
        'ai_absence_of_hallucinations',
        'ai_toxic_language_presence'
    ]

    human_columns = [
        'evaluate the transcription quality',
        'evaluate the model generated summary over the agent notes',
        'evaluate the model generated summary coherence',
        'indicate the presence of hallucinations in the model summary',
        'severity of the hallucination'
    ] + [col for col in encoded_df.columns if col.startswith(tuple(categorical_columns))]

    # Calculate correlations
    correlation_df = calculate_correlations(encoded_df, ai_columns, human_columns)

    # Create correlation matrix for heatmap
    corr_matrix = encoded_df[ai_columns + human_columns].corr(method='spearman')

    # Create and save the correlation heatmap
    heatmap_path = "correlation_heatmap.png"
    create_correlation_heatmap(corr_matrix, heatmap_path)

    # Save the results to the same Excel file with multiple sheets
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        encoded_df.to_excel(writer, sheet_name='Evaluated Data', index=False)
        correlation_df.to_excel(writer, sheet_name='Correlations', index=False)

    print(f"Evaluation complete. Results and correlations saved to '{file_path}'")
    print(f"Correlation heatmap saved as '{heatmap_path}'")

if __name__ == "__main__":
    main()

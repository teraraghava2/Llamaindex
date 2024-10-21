import pandas as pd
import numpy as np
import openai
from typing import List, Dict
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Excel file containing call transcripts and summaries."""
    return pd.read_excel(file_path)

def generate_evaluation(transcript: str, summary: str) -> Dict[str, int]:
    """Generate evaluation scores using GPT model."""
    prompt = f"""
    Given the following call transcript and summary, evaluate on a scale of 1-5 (1 being lowest, 5 being highest) for each of the following criteria:

    1. Transcription Quality
    2. Summary Quality
    3. Summary Coherence
    4. Resolution Capture
    5. Emotional Tone

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
    """Process the dataframe and add evaluation scores."""
    evaluations = []

    for _, row in df.iterrows():
        evaluation = generate_evaluation(row['call_transcript'], row['summary'])
        evaluations.append(evaluation)

    # Convert the list of dictionaries to a DataFrame
    eval_df = pd.DataFrame(evaluations)

    # Rename columns to match the desired label names
    eval_df.columns = [
        'evaluate_transcription_quality',
        'evaluate_summary_quality',
        'evaluate_summary_coherence',
        'resolution_captured_in_summary',
        'emotional_tone_in_summary'
    ]

    # Concatenate the original DataFrame with the evaluation DataFrame
    return pd.concat([df, eval_df], axis=1)

def calculate_correlations(df: pd.DataFrame, ai_columns: List[str], human_columns: List[str]) -> pd.DataFrame:
    """Calculate multiple correlation metrics between AI and human annotations."""
    correlations = []

    for ai_col, human_col in zip(ai_columns, human_columns):
        pearson, _ = pearsonr(df[ai_col], df[human_col])
        spearman, _ = spearmanr(df[ai_col], df[human_col])
        kendall, _ = kendalltau(df[ai_col], df[human_col])

        correlations.append({
            'AI Column': ai_col,
            'Human Column': human_col,
            'Pearson Correlation': pearson,
            'Spearman Correlation': spearman,
            'Kendall Tau Correlation': kendall
        })

    return pd.DataFrame(correlations)

def create_correlation_heatmap(df: pd.DataFrame, ai_columns: List[str], human_columns: List[str], output_path: str):
    """Create and save a heatmap of correlations between AI and human annotations."""
    # Calculate the correlation matrix
    corr_matrix = df[ai_columns + human_columns].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

    # Customize the plot
    plt.title('Correlation Heatmap: AI Evaluations vs Human Annotations')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    # Load the data
    file_path = "path_to_your_excel_file.xlsx"
    df = load_data(file_path)

    # Process the data and get evaluation scores
    result_df = process_data(df)

    # Define the columns for correlation analysis
    ai_columns = [
        'evaluate_transcription_quality',
        'evaluate_summary_quality',
        'evaluate_summary_coherence',
        'resolution_captured_in_summary',
        'emotional_tone_in_summary'
    ]

    human_columns = [
        'evaluate the transcription quality',
        'evaluate the agent notes quality',
        'evaluate the model generated summary coherence',
        'did you think the resolution was captured in the summary',
        'indicate the presence of elation in the model summary'
    ]

    # Calculate correlations
    correlation_df = calculate_correlations(result_df, ai_columns, human_columns)

    # Create and save the correlation heatmap
    heatmap_path = "correlation_heatmap.png"
    create_correlation_heatmap(result_df, ai_columns, human_columns, heatmap_path)

    # Save the results to the same Excel file with multiple sheets
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        result_df.to_excel(writer, sheet_name='Evaluated Data', index=False)
        correlation_df.to_excel(writer, sheet_name='Correlations', index=False)

    print(f"Evaluation complete. Results and correlations saved to '{file_path}'")
    print(f"Correlation heatmap saved as '{heatmap_path}'")

if __name__ == "__main__":
    main()

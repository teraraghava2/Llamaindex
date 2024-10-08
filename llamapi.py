from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import os
from generate_report import generate_report  # Import the function from generate_report.py

app = FastAPI()

# Directory for saving reports
REPORTS_DIR = "reports"

# Ensure the reports directory exists
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

@app.post("/upload/")
async def upload_excel(file: UploadFile = File(...), model_path: str = ""):
    """
    API endpoint to accept an Excel file and process it for generating evaluation reports.
    Input:
        - file: Excel file with columns 'call_transcript' and 'summarizer'
        - model_path: Path to the custom model (for BERTScore or other custom models)
    Output:
        - JSON response with the evaluation metrics for each record
    """
    # Read the uploaded Excel file
    contents = await file.read()
    
    # Save the Excel file temporarily
    temp_excel_path = f"temp_{file.filename}"
    with open(temp_excel_path, 'wb') as f:
        f.write(contents)
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(temp_excel_path)
    
    # Prepare the output JSON structure
    results = []

    # Loop over each record in the DataFrame
    for idx, row in df.iterrows():
        transcript = row['call_transcript']
        summary = row['summarizer']

        # Call the generate_report function for each record
        report = generate_report(transcript, summary, model_path)
        
        # Append the result with record ID
        result = {
            "record_id": idx,
            "metrics": report
        }
        
        results.append(result)
    
    # Delete the temporary Excel file after processing
    os.remove(temp_excel_path)

    # Save the results as a JSON report in the reports folder
    report_path = os.path.join(REPORTS_DIR, f"report_{file.filename.split('.')[0]}.json")
    with open(report_path, 'w') as f:
        f.write(str(results))
    
    # Return the results as a JSON response
    return JSONResponse(content={"status": "success", "data": results})

# For testing purposes, running Uvicorn from this script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
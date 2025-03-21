import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Load environment variables (optional, in case you want to keep the key in .env)
load_dotenv()

# Set your Gemini API Key (replace this if you don't want to use .env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE"))

def generate_schema(user_prompt):
    system_prompt = """
You are an expert data scientist helping users design synthetic datasets.
Given a description of a dataset, generate:
- A list of columns.
- The data type for each column (string, int, float, date).
- Approximate number of rows.

The response should be **pure JSON** like this:
{
    "columns": ["PatientID", "Age", "Gender", "Diagnosis"],
    "types": ["int", "int", "string", "string"],
    "size": 500
}
Only output the JSON - no extra text.
"""

    full_prompt = system_prompt + "\n\nUser request: " + user_prompt

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(full_prompt)

    # Extract and parse the JSON response
    schema = json.loads(response.text.strip())
    return schema

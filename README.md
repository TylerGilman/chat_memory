Chat Memory Fine Tuned llama3.1 8B
Tyler Gilman, George Bikhazi
Project Overview
This project aims to create a fine-tuned LLaMA 3.1 8B model for chat memory and summarization. The process involves data preprocessing, training, and encoding to enable the model to understand and generate temporal summaries of conversations.
File Descriptions

parse_and_generate.py: Main script that:

Loads conversations from CSV
Uses Claude API to expand conversations
Adds timestamps and generates related exchanges
Creates summaries
Saves in both JSON and finetuning formats


.env: Configuration file containing:
CopyANTHROPIC_API_KEY=your_api_key_here
INPUT_CSV=preprocessing/input/no_timestamps.csv
NUM_ROWS_TO_PROCESS=2

Original dataset: https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum (MIT License)
input/no_timestamps.csv: Original conversation data with columns:

id: Unique identifier
dialogue: Conversation text
summary: Brief summary
topic: Main topic



Fine tuning outputs are in chat_memory/preprocessing/transformed_conversations/success
Setup and Installation

Create preprocessing directory and move files:

bashCopymkdir -p chat_memory/preprocessing/input
cd chat_memory/preprocessing

Create and activate virtual environment:

bashCopypython -m venv .trainvenv
source .trainvenv/bin/activate  # On Windows: .trainvenv\Scripts\activate

Install requirements:

bashCopypip install pandas anthropic python-dotenv

Configure .env file (see format above)

Running the Preprocessor

Ensure you're in the preprocessing directory:

bashCopycd chat_memory/preprocessing

Run the script:

bashCopypython parse_and_generate.py
Output Files
The script generates two types of files for each conversation:

Original Format (success/conversation_[id]_original.json):

jsonCopy{
  "original_id": "train_1",
  "topic": "vaccines",
  "original_summary": "Original summary...",
  "transformed_conversations": {
    "today_date": "2024-05-15",
    "personal_summary": "Detailed summary...",
    "conversations": [...]
  }
}

Finetuning Format (success/conversation_[id]_finetune.txt):

Copy[START DATE]
2024-01-15
[END DATE]
2024-02-15
[CHAT MESSAGES]
2024-01-15 09:23 | Person1: message
2024-01-15 09:45 | Person2: message
...
[SUMMARY]
Detailed summary of the conversation history
Model Evaluation
The evaluation process uses two key components:
ROUGE Score Analysis (rouge.py)

Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics
Compares generated summaries against:

Enhanced test data summaries
ChatGPT-generated summaries


Provides detailed statistics including precision, recall, and F1 scores

API Testing (test_api.py)

Tests the deployed model endpoint
Processes test cases from JSONL files
Streams generated summaries
Compares against reference summaries

Generated Data
Success Directory
Contains successfully processed conversations:

conversation_[id]_original.json: Original conversation with metadata
conversation_[id]_finetune.txt: Formatted data for fine-tuning
training_data.jsonl: Combined training data in JSONL format

Errors Directory
Contains failed processing attempts with error details for debugging
Configuration
The system uses environment variables for configuration:

ANTHROPIC_API_KEY: Required for Claude API access
INPUT_CSV: Path to input conversation data
NUM_ROWS_TO_PROCESS: Number of conversations to process
OUTPUT_DIR: Directory for processed data

Model Performance
The system shows strong performance metrics:

ROUGE-1 F1 score of 0.4818 (approaching SOTA performance)
Balanced precision-recall ratios
Natural language flow and structural coherence
Effective summarization strategies aligned with larger models

# Chat Memory Fine Tuned llama3.1 8B

Tyler Gilman, George Bikhazi

## Project Overview
This project aims to create a fine-tuned LLaMA 3.1 8B model for chat memory and summarization. The process involves data preprocessing, training, and encoding to enable the model to understand and generate temporal summaries of conversations.

## Project Structure
```
chat_memory/
├── preprocessing/
│   ├── parse_and_generate.py    # Main preprocessing script
│   ├── .env                     # Configuration file
│   ├── requirements.txt         # Python dependencies
│   ├── input/
│   │   └── no_timestamps.csv    # Original conversation data
│   └── transformed_conversations/
│       ├── success/             # Successful transformations
│       │   ├── *_finetune.txt  # Training-ready format files (committed)
│       │   └── *_original.json # Complete data files (not committed)
│       └── errors/             # Failed transformations
├── training/
│   └── [TODO: Training files]
├── encoding/
│   └── [TODO: Encoding files]
└── server/
    └── [TODO: Server files]
```

## Preprocessing Stage

### File Descriptions
- `parse_and_generate.py`: Main script that:
  - Loads conversations from CSV
  - Uses Claude API to expand conversations
  - Adds timestamps and generates related exchanges
  - Creates summaries
  - Saves in both JSON and finetuning formats

- `.env`: Configuration file containing:
  ```
  ANTHROPIC_API_KEY=your_api_key_here
  INPUT_CSV=preprocessing/input/no_timestamps.csv
  NUM_ROWS_TO_PROCESS=2
  ```
- Original dataset: \ `https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum` (MIT License)
- `input/no_timestamps.csv`: Original conversation data with columns:
  - `id`: Unique identifier
  - `dialogue`: Conversation text
  - `summary`: Brief summary
  - `topic`: Main topic

## Fine tuning outputs are in chat_memory/preprocessing/transformed_conversations/sucess

### Setup and Installation
1. Create preprocessing directory and move files:
```bash
mkdir -p chat_memory/preprocessing/input
cd chat_memory/preprocessing
```

2. Create and activate virtual environment:
```bash
python -m venv .trainvenv
source .trainvenv/bin/activate  # On Windows: .trainvenv\Scripts\activate
```

3. Install requirements:
```bash
pip install pandas anthropic python-dotenv
```

4. Configure .env file (see format above)

### Running the Preprocessor
1. Ensure you're in the preprocessing directory:
```bash
cd chat_memory/preprocessing
```

2. Run the script:
```bash
python parse_and_generate.py
```

### Output Files
The script generates two types of files for each conversation:

1. **Original Format** (`success/conversation_[id]_original.json`):
```json
{
  "original_id": "train_1",
  "topic": "vaccines",
  "original_summary": "Original summary...",
  "transformed_conversations": {
    "today_date": "2024-05-15",
    "personal_summary": "Detailed summary...",
    "conversations": [...]
  }
}
```

2. **Finetuning Format** (`success/conversation_[id]_finetune.txt`):
```
[START DATE]
2024-01-15
[END DATE]
2024-02-15
[CHAT MESSAGES]
2024-01-15 09:23 | Person1: message
2024-01-15 09:45 | Person2: message
...
[SUMMARY]
Detailed summary of the conversation history
```

[Rest of README continues with Training, Encoding stages, etc.]

## Directory Migration Guide
If you already have the preprocessing files in the root directory:

1. Create the preprocessing directory:
```bash
mkdir -p chat_memory/preprocessing/input
```

2. Move existing files:
```bash
mv parse_and_generate.py chat_memory/preprocessing/
mv .env chat_memory/preprocessing/
mv no_timestamps.csv chat_memory/preprocessing/input/
mv transformed_conversations/ chat_memory/preprocessing/
```

3. Update paths in .env file to reflect new structure

## Next Steps
1. Complete preprocessing data generation
2. Move to training stage (see Training section)
3. Develop server integration (see Server section)


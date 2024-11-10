import json
import os
from sklearn.model_selection import train_test_split

def prepare_data():
    """Split data into train and test sets"""
    # Create directories
    for dir_path in ['data/train', 'data/test', 'data/raw']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Move original data to raw if it exists
    if os.path.exists('training_data.json'):
        os.rename('training_data.json', 'data/raw/original_data.jsonl')
    
    # Load data
    with open('data/raw/original_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Save splits
    with open('data/train/train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open('data/test/test.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Data split complete:")
    print(f"Training examples: {len(train_data)}")
    print(f"Testing examples: {len(test_data)}")

if __name__ == "__main__":
    prepare_data()

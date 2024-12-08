# -*- coding: utf-8 -*-
"""train_chat_memory.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pBis1389WNieB0PvsCrLiFBnkWfNM8_c

# Memory-Optimized Chat Memory Model Training

This notebook trains a chat summarization model using the Llama architecture with memory optimizations for Google Colab.

## Setup Steps:
1. Mount Google Drive
2. Install dependencies
3. Configure memory settings
4. Train model with optimizations
5. Save results
"""

# Commented out IPython magic to ensure Python compatibility.
# # Install required packages
# %%capture
# !pip install torch transformers datasets accelerate bitsandbytes trl peft

# Mount Google Drive and setup directories
from google.colab import drive
drive.mount('/content/drive')

# Create project directories
!mkdir -p "/content/drive/MyDrive/chat_memory/models"
!mkdir -p "/content/drive/MyDrive/chat_memory/data"
!mkdir -p "/content/drive/MyDrive/chat_memory/notebooks"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
from tqdm import tqdm
import gc
from datetime import datetime
from datasets import Dataset
from peft import PeftModel


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_model_and_tokenizer():
    print("Setting up model and tokenizer...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def prepare_training_data(data_path, tokenizer, max_length=1024, validation_split=0.1):
    print("Loading and preparing training data...")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Split data into train and validation
    val_size = int(len(data) * validation_split)
    train_data = data[:-val_size]
    val_data = data[-val_size:]

    print(f"Total examples: {len(data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    def format_data(examples, desc):
        formatted_data = []
        for item in tqdm(examples, desc=desc):
            messages_text = "\n".join([
                f"{msg['timestamp']} | {msg['content']}"
                for msg in item['messages']
            ])

            instruction = """You are a chat summarization assistant. Create a CONCISE summary of the following conversation. Focus on key events, decisions, and relationship developments. Be specific about dates and events. DO NOT add commentary or reflections. End your summary with [/SUMMARY]."""

            input_text = f"""[START DATE]
{item['start_date']}
[END DATE]
{item['end_date']}
[CHAT MESSAGES]
{messages_text}"""

            # Ensure the summary ends with the tag
            summary = item['summary'].strip()
            if not summary.endswith('[/SUMMARY]'):
                summary = f"{summary} [/SUMMARY]"

            prompt = f"""<s>[INST] {instruction}

{input_text} [/INST]
[SUMMARY]
{summary}{tokenizer.eos_token}"""

            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )

            formatted_data.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"].copy()
            })

            if len(formatted_data) == 1 and desc == "Processing training examples":
                print("\nSample prompt length:", len(encoded["input_ids"]))
                print("Sample prompt preview:")
                print(prompt[:500] + "...")

        return Dataset.from_list(formatted_data)

    train_dataset = format_data(train_data, "Processing training examples")
    val_dataset = format_data(val_data, "Processing validation examples")

    return train_dataset, val_dataset

def train_model():
    base_dir = "/content/drive/MyDrive/chat_memory"
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)

    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Explicitly set padding side
    tokenizer.padding_side = 'right'

    # Properly prepare model for training - simplified parameters
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    # Configure LoRA with adjusted parameters
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # Prepare datasets
    train_dataset, val_dataset = prepare_training_data(
        f"{base_dir}/data/train.jsonl",
        tokenizer,
        max_length=1024
    )

    # Training arguments with adjusted parameters
    training_args = TrainingArguments(
        output_dir=f"{base_dir}/models/training_checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=5e-5,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1,
        weight_decay=0.05,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        save_steps=200,
        eval_steps=200,
        save_total_limit=5,
        logging_dir=f"{base_dir}/logs",
        report_to="none",
        remove_unused_columns=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Add these parameters
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        torch_compile=False,  # Disable torch compile
        gradient_checkpointing_kwargs={"use_reentrant": False}  # Important for stability
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        packing=False
    )

    print("Starting training...")
    try:
        # Enable gradients explicitly
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: grad.clone())

        trainer.train()

        print("Saving PEFT model...")
        trainer.model.save_pretrained(f"{base_dir}/models/peft_model")

        print("Merging and saving final model...")
        # Clear everything before merging
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Set memory efficiency environment variable
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            device_map="auto",  # Use automatic device mapping
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        print("Loading PEFT model...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            f"{base_dir}/models/peft_model",
            device_map="auto",
            torch_dtype=torch.float16,
            is_trainable=False
        )

        print("Merging models...")
        try:
            merged_model = peft_model.merge_and_unload()
            print("Merge successful!")

            print("Saving merged model...")
            merged_model_path = f"{base_dir}/models/merged_model"
            merged_model.save_pretrained(
                merged_model_path,
                safe_serialization=True,
                max_shard_size="2GB"  # Save in smaller chunks
            )
            tokenizer.save_pretrained(merged_model_path)
            print("Merged model saved successfully!")

        except Exception as merge_error:
            print(f"Error during merge: {str(merge_error)}")
            print("Saving PEFT model only...")
            # Save the PEFT model configuration and states
            peft_model.save_pretrained(
                f"{base_dir}/models/peft_model_backup",
                safe_serialization=True
            )
            print("PEFT model saved as backup")
            raise merge_error

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise e

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()



train_model()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
import os

def merge_and_test_model():
    # Clear memory first
    gc.collect()
    torch.cuda.empty_cache()

    base_dir = "/content/drive/MyDrive/chat_memory"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        print("Loading PEFT model...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            f"{base_dir}/models/peft_model",
            device_map="auto",
            torch_dtype=torch.float16,
            is_trainable=False
        )

        print("Merging models...")
        merged_model = peft_model.merge_and_unload()

        # Clear memory after merging
        del base_model
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()

        print("Saving merged model...")
        merged_model_path = f"{base_dir}/models/merged_model"
        merged_model.save_pretrained(
            merged_model_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        tokenizer.save_pretrained(merged_model_path)

        print("Testing merged model...")
        test_prompt = """<s>[INST] You are a chat summarization assistant. Please succinctly summarize this conversation:

[START DATE]
2024-01-01
[END DATE]
2024-01-01
[CHAT MESSAGES]
2024-01-01 10:00 | User A: How was your weekend?
2024-01-01 10:05 | User B: It was amazing! I finally went to that new restaurant downtown.
2024-01-01 10:06 | User A: Oh nice! How was the food?
2024-01-01 10:08 | User B: Absolutely delicious! Their pasta is to die for. We should go together sometime! [/INST]
[SUMMARY]"""

        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to("cuda")

        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            num_beams=5,
            no_repeat_ngram_size=3,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nTest Generation:")
        print("-" * 50)
        print(response)
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

# Run the merge and test
merge_and_test_model()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
import gc
from datetime import datetime

def evaluate_model():
    base_dir = "/content/drive/MyDrive/chat_memory"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        f"{base_dir}/models/merged_model",
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/models/merged_model")

    print("Loading test data...")
    with open(f"{base_dir}/data/test.jsonl", 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    print(f"Loaded {len(test_data)} test examples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_dir}/results/evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Open files with UTF-8 encoding
    with open(f"{results_dir}/all_results.txt", 'w', encoding='utf-8') as f_txt, \
         open(f"{results_dir}/all_results.jsonl", 'w', encoding='utf-8') as f_jsonl:

        f_txt.write("MODEL EVALUATION RESULTS\n")
        f_txt.write(f"Date: {timestamp}\n")
        f_txt.write(f"Number of test examples: {len(test_data)}\n\n")

        for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
            messages_text = "\n".join([
                f"{msg['timestamp']} | {msg['content']}"
                for msg in item['messages']
            ])

            prompt = f"""<s>[INST] You are a chat summarization assistant. Please summarize this conversation:

[START DATE]
{item['start_date']}
[END DATE]
{item['end_date']}
[CHAT MESSAGES]
{messages_text} [/INST]
[SUMMARY]"""

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                num_beams=5,
                no_repeat_ngram_size=3,
                do_sample=True
            )

            # Ensure proper string handling
            try:
                generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

                f_txt.write(f"\nExample {idx + 1} (ID: {item.get('id', f'test_{idx}')})\n")
                f_txt.write("="*80 + "\n")
                f_txt.write("Conversation Period:\n")
                f_txt.write(f"From: {item['start_date']} To: {item['end_date']}\n\n")
                f_txt.write("Original Summary:\n")
                f_txt.write(item['summary'] + "\n\n")
                f_txt.write("Generated Summary:\n")
                f_txt.write(generated_summary + "\n")
                f_txt.write("="*80 + "\n")

                result = {
                    'id': item.get('id', f'test_{idx}'),
                    'start_date': item['start_date'],
                    'end_date': item['end_date'],
                    'original_summary': item['summary'],
                    'generated_summary': generated_summary,
                    'num_messages': len(item['messages'])
                }
                f_jsonl.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"\nError processing example {idx}: {str(e)}")
                continue

            if (idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print("\nEvaluation completed!")
    print(f"Results saved in {results_dir}")
    print(f"- Detailed results: {results_dir}/all_results.txt")
    print(f"- JSON format: {results_dir}/all_results.jsonl")

    del model
    gc.collect()
    torch.cuda.empty_cache()

# Run evaluation
evaluate_model()


import json
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gc
import os

# Configuration
max_seq_length = 4096  # Increased for better context handling
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples['messages'])):
        messages_text = "\n".join([
            f"{msg['timestamp']} | {msg['content']}"
            for msg in examples['messages'][i]
        ])

        instruction = "Summarize this chat conversation."

        prompt = (
            f"<s>[INST] {instruction}\n\n"
            f"[START DATE]\n{examples['start_date'][i]}\n"
            f"[END DATE]\n{examples['end_date'][i]}\n"
            f"[CHAT MESSAGES]\n{messages_text} [/INST]\n"
            f"[SUMMARY]\n{examples['summary'][i]}</s>{tokenizer.eos_token}"
        )

        texts.append(prompt)
    return {"text": texts}

def load_and_prepare_data(file_path):
    # Load the JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Create dataset
    dataset = Dataset.from_list(data)
    return dataset

def main():
    global tokenizer
    clear_memory()

    # BitsAndBytes configuration for 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    clear_memory()

    # Load model with 8-bit quantization
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    clear_memory()

    # Apply LoRA with increased rank for better capacity
    lora_config = LoraConfig(
        r=64,  # Increased from 4 to 64
        lora_alpha=32,  # Adjusted lora_alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    clear_memory()

    # Load data
    print("Loading data...")
    dataset = load_and_prepare_data('data/train.jsonl')
    clear_memory()

    # Format dataset
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=8,  # Increased batch size
        num_proc=4,    # Use multiprocessing if available
        remove_columns=dataset.column_names
    )
    clear_memory()

    # Training arguments optimized for performance
    training_args = TrainingArguments(
        output_dir="./models/chat_memory_model",
        per_device_train_batch_size=4,  # Increased batch size
        gradient_accumulation_steps=4,  # Adjusted for effective batch size
        max_steps=1000,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    # Disable model caching
    model.config.use_cache = False

    print("Starting training...")
    trainer_stats = trainer.train()

    print("\nTraining completed!")
    print(f"Training time: {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes")

    # Save the model
    print("Saving model...")
    model.save_pretrained("models/chat_memory_model")
    tokenizer.save_pretrained("models/chat_memory_model")

if __name__ == "__main__":
    main()

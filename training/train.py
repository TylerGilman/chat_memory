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
from peft import LoraConfig, get_peft_model
import gc

# Configuration
max_seq_length = 4096  # Increased for longer conversations
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Clear CUDA cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    rope_scaling={
        "type": "llama3",
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# Load JSONL data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# Load your data
chat_data = load_jsonl("your_data.jsonl")

# Create dataset
dataset = Dataset.from_list(chat_data)


def formatting_prompts_func(examples):
    texts = []
    for item in examples:
        # Format the messages
        messages_text = "\n".join(
            [f"{msg['timestamp']} | {msg['content']}" for msg in item["messages"]]
        )

        # Create the instruction prompt
        instruction = """You are a chat summarization assistant. Given a conversation and its date range, create a concise yet comprehensive summary that captures the key points, emotional undertones, and progression of the relationship between participants. Pay attention to significant events, changes in dynamics, and important developments over time."""

        # Format the input with context
        input_text = f"""Please summarize the following chat conversation that occurred between {item['start_date']} and {item['end_date']}.

[START DATE]
{item['start_date']}
[END DATE]
{item['end_date']}
[CHAT MESSAGES]
{messages_text}"""

        # Create the full prompt with instruction format
        prompt = f"""<s>[INST] {instruction}

{input_text} [/INST]
[SUMMARY]
{item['summary']}</s>{tokenizer.eos_token}"""

        texts.append(prompt)
    return {"text": texts}


# Apply formatting to dataset
formatted_dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    batch_size=1,
    remove_columns=dataset.column_names,
)

# Set up the trainer
training_args = TrainingArguments(
    output_dir="./chat_memory_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=500,  # Adjust based on your needs
    learning_rate=1e-4,
    fp16=True,
    logging_steps=1,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="constant",
    save_steps=100,
    save_total_limit=2,
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

print(
    f"GPU memory allocated before training: {torch.cuda.memory_allocated(0)/1e9:.2f} GB"
)
print(
    f"GPU memory reserved before training: {torch.cuda.memory_reserved(0)/1e9:.2f} GB"
)

# Train the model
trainer_stats = trainer.train()

# Print training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")

# Save the model
model.save_pretrained("chat_memory_model")
tokenizer.save_pretrained("chat_memory_model")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Clear memory cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Initialzie 
model = AutoModelForCausalLM.from_pretrained('/deepseek_model', torch_dtype=torch.float16, device_map='auto')
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained('/deepseek_tokenizer')
dataset = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en')['train']

'''
 Tokenize text, use chain of thought (cot) text to guide generation of response
'''
def tokenize(example):

    # Prompt structure
    prompt = tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content': 'You are a medical expert helping patients by answering their medical questions.'},
            {'role': 'user', 'content': example['Question']}

        ],
        tokenize=False,
        add_generation_prompt=True
    )

    # Use complex chain of though to help model with reasoning
    cot = f"Think step by step: \n{example['Complex_CoT']}\n\n"

    response = f"Final response: {example['Response']}"

    full_text = prompt + cot + response

    tokenized_text = tokenizer(
        full_text,
        padding='max_length',
        truncation=True,
        max_length=1012,
        return_tensors="pt",
    )

    # Label mask 
    input_ids = tokenized_text["input_ids"][0]
    labels = input_ids.clone()

    # Calculate where to start predicting
    cot_len = len(tokenizer(cot, add_special_tokens=False)["input_ids"])
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    start_of_label = prompt_len + cot_len

    # Mask everything before the start of the response
    labels[:start_of_label] = -100
    tokenized_text["labels"] = labels

    return {
        "input_ids": tokenized_text["input_ids"][0],
        "attention_mask": tokenized_text["attention_mask"][0],
        "labels": tokenized_text["labels"]
    }


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./deepseek-medchat",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
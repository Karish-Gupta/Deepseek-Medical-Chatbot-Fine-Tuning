from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from model_training import Model

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
dataset = load_dataset('FreedomIntelligence/Medical-R1-Distill-Data')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize_function(text):
   return tokenizer(text['question'], text['reasoning (reasoning_content)'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train and test splits
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=123)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Utilize cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name)
deepseek_model.to(device)

fine_tuned_model = Model(
   model=deepseek_model, 
   tokenizer=tokenizer, 
   train_dataset=train_dataset, 
   test_dataset=test_dataset, 
   lr=5e-5, 
   num_epochs=3, 
   weight_decay=0.01, 
   batch_size=2, 
)

# Save model
fine_tuned_model.model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

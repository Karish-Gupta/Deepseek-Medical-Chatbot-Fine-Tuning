import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load saved model
model_path = './fine_tuned_model'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# sample_question = "Based on a physical examination revealing a pulsatile abdominal mass at the junction of the periumbilical and suprapubic regions, combined with a patient history of hypertension, peripheral vascular disease, and smoking, and the presence of an abdominal bruit, what is the most likely diagnosis?"

sample_question = "What is 9 + 10"


# Tokenize the input
inputs = tokenizer(sample_question, return_tensors="pt", padding=True, truncation=True)

# Predict

with torch.no_grad():
   outputs = model(**inputs)

logits = outputs.logits
predicted_ids = torch.argmax(logits, dim=-1)
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
print(predicted_text)
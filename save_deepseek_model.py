from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained('/deepseek_model')
tokenizer.save_pretrained('/deepseek_tokenizer')


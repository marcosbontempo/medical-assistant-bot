import transformers
import torch

# Fetch model from Hugging Face
model_id = "aaditya/OpenBioLLM-Llama3-8B"
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Save model and tokenizer locally
model_directory = "../models/reference/"
model.save_pretrained(model_directory)
tokenizer.save_pretrained(model_directory)

print(f"Model and Tokenizer saved in: {model_directory}")
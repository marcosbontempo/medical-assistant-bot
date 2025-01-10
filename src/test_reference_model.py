import transformers
import torch

model_id = "aaditya/OpenBioLLM-Llama3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Prompt context and question
messages = [
    {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
    {"role": "user", "content": "What are the symptoms of Diabetes?"}
]

# Prompt building
prompt = ""
for message in messages:
    prompt += f"{message['role'].capitalize()}: {message['content']}\n"

outputs = pipeline(
    prompt,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    eos_token_id=pipeline.tokenizer.eos_token_id,
)

print(outputs[0]["generated_text"][len(prompt):])

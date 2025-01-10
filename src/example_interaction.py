import transformers
import torch
import pandas as pd

# Fine-tuned model path
model_id = "../models/finetune/" 

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Load dataset to select 10 example questions
csv_path = "../data/raw/intern_screening_dataset.csv"
df = pd.read_csv(csv_path)

questions_indexes = [1, 13, 62, 91, 118, 228, 250, 302, 443, 612]

for idx in questions_indexes:
    question = df.iloc[idx]['question']
    
    messages = [
        {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
        {"role": "user", "content": question}
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
    
    generated_text = outputs[0]['generated_text'][len(prompt):]
    
    # Remove OpenBioLLM from answer
    generated_text = generated_text.replace("OpenBioLLM:", "").strip()

    print(f"Question: {question}")
    print(f"Answer: {generated_text}")
    print("-" * 50)  

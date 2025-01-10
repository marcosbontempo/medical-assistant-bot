import transformers
import torch
import pandas as pd
from rouge_score import rouge_scorer

finetuned_model_id = "../models/finetune/"  
reference_model_id = "aaditya/OpenBioLLM-Llama3-8B"

# Reference and fine-tuned models pipelines
finetuned_pipeline = transformers.pipeline(
    "text-generation",
    model=finetuned_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

reference_pipeline = transformers.pipeline(
    "text-generation",
    model=reference_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Loading dataset and selecting 10 example questions
csv_path = "../data/raw/intern_screening_dataset.csv"
df = pd.read_csv(csv_path)

questions_indexes = [1, 13, 62, 91, 118, 228, 250, 302, 443, 612]

# Initializing ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_scores_finetuned = []
rouge_scores_reference = []

# Iterating over example questions
for idx in questions_indexes:
    question = df.iloc[idx]['question']    
    answer_from_dataset = df.iloc[idx]['answer']
    
    messages = [
        {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."},
        {"role": "user", "content": question}
    ]
    
    # Prompt building
    prompt = ""
    for message in messages:
        prompt += f"{message['role'].capitalize()}: {message['content']}\n"
    
    # Fine-tuned model output
    finetuned_outputs = finetuned_pipeline(
        prompt,
        max_new_tokens=512,  
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        eos_token_id=finetuned_pipeline.tokenizer.eos_token_id,
    )
    
    # Reference model output
    reference_outputs = reference_pipeline(
        prompt,
        max_new_tokens=512,  
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=reference_pipeline.tokenizer.eos_token_id,
    )
    
    finetuned_generated_text = finetuned_outputs[0]['generated_text'][len(prompt):]
    reference_generated_text = reference_outputs[0]['generated_text'][len(prompt):]
    
    # Removing OpenBioLLM from answer
    finetuned_generated_text = finetuned_generated_text.replace("OpenBioLLM:", "").strip()
    reference_generated_text = reference_generated_text.replace("OpenBioLLM:", "").strip()

    # Calculating ROUGE
    finetuned_rouge_score = scorer.score(answer_from_dataset, finetuned_generated_text)
    reference_rouge_score = scorer.score(answer_from_dataset, reference_generated_text)

    rouge_scores_finetuned.append(finetuned_rouge_score)
    rouge_scores_reference.append(reference_rouge_score)

    print(f"Question: {question}")
    print(f"Answer from Dataset: {answer_from_dataset}")
    print(f"Answer from Fine-tuned Model: {finetuned_generated_text}")
    print(f"Answer from Reference Model: {reference_generated_text}")
    print("-" * 50)  

# Printing ROUGE metrics
def average_rouge(rouge_scores):
    rouge1_f1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
    rouge2_f1 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
    rougeL_f1 = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
    return {'rouge1_f1': rouge1_f1, 'rouge2_f1': rouge2_f1, 'rougeL_f1': rougeL_f1}

avg_rouge_finetuned = average_rouge(rouge_scores_finetuned)
avg_rouge_reference = average_rouge(rouge_scores_reference)

print("Average ROUGE scores for Fine-tuned Model:", avg_rouge_finetuned)
print("Average ROUGE scores for Reference Model:", avg_rouge_reference)

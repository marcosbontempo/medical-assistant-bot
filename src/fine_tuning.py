import json
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk, Dataset
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from evaluate import load as load_metric

# Load preprocessed dataset
tokenized_datasets = load_from_disk('../data/processed')
tokenized_datasets = tokenized_datasets.map(lambda x: {'id': list(range(len(x)))})
train_test_split = tokenized_datasets.train_test_split(test_size=0.3)

# Loading reference model
model_id = "aaditya/OpenBioLLM-Llama3-8B"  
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# GPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  

# LoRA initialization
lora_config = LoraConfig(
    r=16,                           
    lora_alpha=32,                  
    lora_dropout=0.1,               
    task_type="QUESTION_ANS"        
)

peft_model = get_peft_model(model, lora_config)

# Initializing training
training_args = TrainingArguments(
    output_dir="./results_qa",       
    evaluation_strategy="epoch",     
    learning_rate=5e-5,              
    per_device_train_batch_size=2,   
    num_train_epochs=3,              
    weight_decay=0.01,               
    logging_dir="./logs",            
    save_total_limit=2,              
    save_strategy="epoch",           
    fp16=False,                       
    logging_steps=50,  
    logging_first_step=True,  
    report_to="tensorboard",  
    disable_tqdm=False,  
    no_cuda=False,  
    remove_unused_columns=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"]
)

trainer.train()

# Saving fine-tuned model
peft_model.save_pretrained('../models/finetune')

# Training and validation loss visualization
training_logs = trainer.state.log_history

train_loss = [log['loss'] for log in training_logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in training_logs if 'eval_loss' in log]

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

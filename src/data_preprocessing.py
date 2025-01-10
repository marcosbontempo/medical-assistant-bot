import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer

# Special characters cleaning
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Punctuation
    text = re.sub(r'\d+', '', text)  # Numbers
    return text

# Get start and end positions of the answer
def get_answer_positions(question, answer):
    context = question + " " + answer
    start = context.find(answer)
    end = start + len(answer) - 1

    # Tokenize the context
    tokenized_context = tokenizer(context, truncation=True, padding='max_length', max_length=512)

    start_token_pos = tokenized_context.char_to_token(start)
    end_token_pos = tokenized_context.char_to_token(end)

    # Get last valid token in case of None
    if start_token_pos is None or end_token_pos is None:
        start_token_pos = len(tokenized_context['input_ids']) - 1
        end_token_pos = len(tokenized_context['input_ids']) - 1

    return start_token_pos, end_token_pos


df = pd.read_csv('../data/raw/intern_screening_dataset.csv')  

# Removed entries with missing answers
df = df.dropna(subset=['answer'])

# Removed duplicate question-answer pairs
df['question_answer'] = df['question'] + ' | ' + df['answer']
df = df.drop_duplicates(subset='question_answer')
df.drop(columns='question_answer', inplace=True)

# Dropped answers shorter than 100 characters
df = df[df['answer'].str.strip().str.len() > 100]

# Normalized text (lowercase and stripped spaces)
df['question'] = df['question'].str.lower()
df['answer'] = df['answer'].str.lower()

df['question'] = df['question'].str.strip()
df['answer'] = df['answer'].str.strip()

# Removed special characters
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# Tokenized questions and answers using Hugging Face AutoTokenizer, creating input_ids and labels
model_id = "aaditya/OpenBioLLM-Llama3-8B"  
tokenizer = AutoTokenizer.from_pretrained(model_id)

df['input_ids'] = df['question'].apply(lambda x: tokenizer.encode(x, padding='max_length', truncation=True, max_length=512))
df['labels'] = df['answer'].apply(lambda x: tokenizer.encode(x, padding='max_length', truncation=True, max_length=512))

# Computed start and end token positions for answers using char_to_token
df['start_positions'], df['end_positions'] = zip(*df.apply(lambda x: get_answer_positions(x['question'], x['answer']), axis=1))

# Drop as colunas question e answer
df.drop(columns=['question', 'answer'], inplace=True)

# Keeping only necessary columns
df = df.reset_index(drop=True)
df = df[['input_ids', 'labels', 'start_positions', 'end_positions']]

print(df.head(5))

# Converted the cleaned DataFrame to a Hugging Face Dataset and saved it
dataset = Dataset.from_pandas(df)
dataset.save_to_disk('../data/processed/intern_screening_dataset')

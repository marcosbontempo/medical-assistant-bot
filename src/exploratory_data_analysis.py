import pandas as pd

df = pd.read_csv('../data/raw/intern_screening_dataset.csv') 

print(f"Total number of entries: {df.shape[0]}")  

print(f"Empty questions: {df['question'].isnull().sum()}")
print(f"Empty answers: {df['answer'].isnull().sum()}")

df['question_answer'] = df['question'] + '|' + df['answer']
duplicate_count = df['question_answer'].duplicated().sum()
print(f"Duplicated question-answer pairs: {duplicate_count}")

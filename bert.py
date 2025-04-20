import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from datasets import Dataset
import numpy as np
from evaluate import load
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Load training and testing data
train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)
test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names)

# Map labels: 1 for 'true', 0 for everything else
train_df['label'] = train_df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
test_df['label'] = test_df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
print(test_df['label'].value_counts())

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply to statement column
train_df['clean_text'] = train_df['statement'].apply(preprocess_text)
test_df['clean_text'] = test_df['statement'].apply(preprocess_text)

train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['clean_text', 'label']])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["clean_text"], truncation=True)

tokenized_train = train_dataset.map(preprocess_function)
tokenized_test = test_dataset.map(preprocess_function)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define evaluation metrics
def compute_metrics(eval_pred):
    accuracy = load("accuracy")
    f1 = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1_score}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate()

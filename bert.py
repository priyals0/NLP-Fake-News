import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from evaluate import load
from transformers import (AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer)

# download resources
nltk.download("punkt")
nltk.download("wordnet")

# col names
column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# load data
train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)
test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names)

# map labels
def map_label(label):
    label = str(label).strip().lower()
    if label in ["true", "mostly-true"]:
        return 1
    elif label in ["false", "pants-fire"]:
        return 0
    else:
        return None

train_df['label'] = train_df['label'].apply(map_label)
test_df['label'] = test_df['label'].apply(map_label)

# drop others
train_df = train_df.dropna(subset=['label'])
test_df = test_df.dropna(subset=['label'])

# make labels ints
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# balance dataset, 50/50
true_df = train_df[train_df['label'] == 1]
false_df = train_df[train_df['label'] == 0]
min_count = min(len(true_df), len(false_df))
balanced_train_df = pd.concat([
    true_df.sample(n=min_count, random_state=42),
    false_df.sample(n=min_count, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# preprocess
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

balanced_train_df['clean_text'] = balanced_train_df['statement'].apply(preprocess_text)
test_df['clean_text'] = test_df['statement'].apply(preprocess_text)

# make dataset
train_dataset = Dataset.from_pandas(balanced_train_df[['clean_text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['clean_text', 'label']])

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["clean_text"], truncation=True)

tokenized_train = train_dataset.map(preprocess_function)
tokenized_test = test_dataset.map(preprocess_function)

# setup model and trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    accuracy = load("accuracy")
    f1 = load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }

# training args
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train and evaluate
train_result = trainer.train()
eval_results = trainer.evaluate()
print(eval_results)

# Confusion Matrix
preds_output = trainer.predict(tokenized_test)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Training/Validation Loss & Accuracy Plot
logs = trainer.state.log_history
train_loss, eval_loss, eval_acc = [], [], []
epochs = []

for log in logs:
    if "loss" in log and "epoch" in log:
        train_loss.append(log["loss"])
        epochs.append(log["epoch"])
    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])
    if "eval_accuracy" in log:
        eval_acc.append(log["eval_accuracy"])

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs[:len(train_loss)], train_loss, label="Train Loss")
plt.plot(epochs[:len(eval_loss)], eval_loss, label="Eval Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs[:len(eval_acc)], eval_acc, label="Eval Accuracy", color="green")
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
import csv
import re
import ssl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# SSL for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

#create label to classification relations
true_labels = ['true', 'mostly-true']
false_labels = ['false', 'pants-fire']

label_mapping = {label: 1 for label in true_labels}
label_mapping.update({label: 0 for label in false_labels})

# Read in training dataset
col_dict = {
    "id": 0, 
    "label": 1, 
    "statement": 2, 
    "subject": 3, 
    "speaker": 4,
    "speaker_job_title": 5,
    "state": 6,
    "party_affiliation": 7, 
    "barely_true_counts": 8,
    "false_counts": 9,
    "half_true_counts": 10,
    "mostly_true_counts": 11, 
    "pants_on_fire_counts": 12,
    "context": 13
}

column_names = ["label", "statement", 'subject', 'speaker', 'speaker_job_title', 'party_affiliation', 'context']
usecols = [col_dict[key] for key in column_names]

train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names, usecols=usecols, quoting=csv.QUOTE_NONE)
test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names, usecols=usecols, quoting=csv.QUOTE_NONE)

# create new col with 'class' being either 0/1 based on the label
# drop any row that doesn't have one of the labels we're looking at
def fixDataFrame(df, label_mapping):
    df['class'] = df['label'].str.strip().str.lower().map(label_mapping)
    return df.dropna(subset=['class'])

train_df = fixDataFrame(train_df, label_mapping)
test_df = fixDataFrame(test_df, label_mapping)

# Create True and Fake media claims df
true_df = train_df[train_df['label'].isin(true_labels)]
fake_df = train_df[train_df['label'].isin(false_labels)]

# max poss claims that can be used to train false and true 
# balance the training dataset (50/50)
maxClaims = min(len(fake_df), len(true_df))
true_df = true_df.sample(maxClaims, random_state=10)
false_df = fake_df.sample(maxClaims, random_state=10)
balanced_train_df = pd.concat([true_df, false_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

balanced_train_df['clean_text'] = balanced_train_df['statement'].apply(preprocess_text)
test_df['clean_text'] = test_df['statement'].apply(preprocess_text)

# Vectorization
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(balanced_train_df['clean_text'])
y_train = balanced_train_df['class']
X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['class']

# Random Baseline
random_clf = DummyClassifier(strategy='uniform', random_state=42)
random_clf.fit(X_train, y_train)
random_pred = random_clf.predict(X_test)

random_accuracy = accuracy_score(y_test, random_pred)
random_precision = precision_score(y_test, random_pred, zero_division=0)
random_recall = recall_score(y_test, random_pred, zero_division=0)
random_f1 = f1_score(y_test, random_pred, zero_division=0)

print("\nRandom Baseline:")
print(f"Accuracy: {random_accuracy:.4f}")
print(f"Precision: {random_precision:.4f}")
print(f"Recall: {random_recall:.4f}")
print(f"F1 Score: {random_f1:.4f}")

# Majority Class Baseline
majority_clf = DummyClassifier(strategy='most_frequent')
majority_clf.fit(X_train, y_train)
majority_pred = majority_clf.predict(X_test)

majority_accuracy = accuracy_score(y_test, majority_pred)
majority_precision = precision_score(y_test, majority_pred, zero_division=0)
majority_recall = recall_score(y_test, majority_pred, zero_division=0)
majority_f1 = f1_score(y_test, majority_pred, zero_division=0)

print("\nMajority Class Baseline:")
print(f"Accuracy: {majority_accuracy:.4f}")
print(f"Precision: {majority_precision:.4f}")
print(f"Recall: {majority_recall:.4f}")
print(f"F1 Score: {majority_f1:.4f}")

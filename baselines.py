import pandas as pd
import numpy as np
import re
import nltk
import ssl
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# SSL for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Col names
column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Load data
train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)
test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names)

# Labels
true_labels = ['true', 'mostly-true']
false_labels = ['false', 'pants-fire']

# Real if "true" or "mostly-true," Fake if "False" or "pants-fire"
train_df['label'] = train_df['label'].apply(lambda x: 1 if str(x).strip().lower() in true_labels else (0 if str(x).strip().lower() in false_labels else np.nan))
test_df['label'] = test_df['label'].apply(lambda x: 1 if str(x).strip().lower() in true_labels else (0 if str(x).strip().lower() in false_labels else np.nan))

# Drop rows with other labels ("half-true" and "barely-true")
train_df = train_df.dropna(subset=['label']).copy()
test_df = test_df.dropna(subset=['label']).copy()

# Labels as ints
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Print label distribution
print("Training label dist:")
print(train_df['label'].value_counts().rename({0: "Fake", 1: "Real"}))
print("\nTest label dist:")
print(test_df['label'].value_counts().rename({0: "Fake", 1: "Real"}))

# Balance the training dataset (50/50)
true_df = train_df[train_df['label'] == 1].sample(n=train_df['label'].value_counts()[0], random_state=42)
false_df = train_df[train_df['label'] == 0]
balanced_train_df = pd.concat([true_df, false_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced training label dist:")
print(balanced_train_df['label'].value_counts().rename({0: "Fake", 1: "Real"}))

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
y_train = balanced_train_df['label']
X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['label']

# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nNaive Bayes:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix for Naive Bayes
nb_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues',
            yticklabels=["Predicted Fake", "Predicted Real"],
            xticklabels=["Actual Fake", "Actual Real"])
plt.title("Naive Bayes Confusion Matrix")
plt.gca().xaxis.set_label_position('top') 
plt.gca().xaxis.tick_top()
plt.tight_layout()
plt.show(block=False)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print("Logistic Regression:")
print(f"Accuracy: {log_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, log_pred))

# Confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, log_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Reds',
            yticklabels=["Predicted Fake", "Predicted Real"],
            xticklabels=["Actual Fake", "Actual Real"])
plt.title("Logistic Regression Confusion Matrix")
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.show(block=False)

# SVM
svm = LinearSVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVC:")
print(f"Accuracy: {svm_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, svm_pred))

# Confusion matrix for SVM
cm_svm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples',
            yticklabels=["Predicted Fake", "Predicted Real"],
            xticklabels=["Actual Fake", "Actual Real"])
plt.title("SVM Confusion Matrix")
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.show(block=False)

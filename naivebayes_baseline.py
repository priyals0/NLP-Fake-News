import pandas as pd
import numpy as np
import re
import nltk
import ssl
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

#need to manually add column names
column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

#reading in the tsvs and saving them into training and testing df
train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)
test_df = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names)
print(train_df.columns.tolist())

#mapping labels 1 if true, 0 if not
train_df['label'] = train_df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
test_df['label'] = test_df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)

#preprocessing the text: tokenizing and lemmatizing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

#working with the statement column
train_df['clean_text'] = train_df['statement'].apply(preprocess_text)
test_df['clean_text'] = test_df['statement'].apply(preprocess_text)



#bag of words approach
vectorizer = CountVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
y_train = train_df['label']
y_test = test_df['label']

#naive bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

#printing out accuracy report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


#the following are for debugg
# === (1) Show Most Informative Features ===
""" def show_top_features(classifier, vectorizer, n=20):
    feature_names = vectorizer.get_feature_names_out()
    class0 = classifier.feature_log_prob_[0]  # fake news
    class1 = classifier.feature_log_prob_[1]  # real news

    top0 = np.argsort(class0)[-n:]  # Top words for fake
    top1 = np.argsort(class1)[-n:]  # Top words for real

    print("\nTop fake news indicators:")
    for i in reversed(top0):
        print(f"{feature_names[i]} ({class0[i]:.4f})")

    print("\nTop real news indicators:")
    for i in reversed(top1):
        print(f"{feature_names[i]} ({class1[i]:.4f})")

show_top_features(nb_classifier, vectorizer)


# === (2) Accuracy with Shuffled Labels (sanity check for data leakage) ===
print("\nTesting with shuffled labels for data leakage check...")
shuffled_y = df['label'].sample(frac=1.0, random_state=42).reset_index(drop=True)
X_all = vectorizer.fit_transform(df['clean_text'])
X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = train_test_split(X_all, shuffled_y, test_size=0.2, random_state=42)

nb_shuffled = MultinomialNB().fit(X_train_shuf, y_train_shuf)
shuf_pred = nb_shuffled.predict(X_test_shuf)
shuf_acc = accuracy_score(y_test_shuf, shuf_pred)
print(f"Accuracy on shuffled labels: {shuf_acc:.4f}")

 """
import nltk
nltk.download('stopwords')
from nltk import FreqDist
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


from sklearn import metrics

stops = stopwords.words('english')
stops.extend([",", ".", "!", "?", "'", '"', "I", "i", "n't", "'ve", "'d", "'s"])

allwords = []

# Read in the training dataset
train_df = pd.read_csv("/content/NLP-Fake-News/train.tsv", sep='\t', header=None)
train_df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                  'state', 'party', 'barely_true', 'false', 'half_true',
                  'mostly_true', 'pants_on_fire', 'context']

# Filter for true news
print("True news articles")
true_df = train_df[train_df['label'].str.lower() == 'true']
true_articles = []

# Read statement and clean
for idx, row in true_df.iterrows():
    if idx == 2000:
        break
    full_text = row['statement']
    full_text = re.sub(r'[^\w\s]', ' ', full_text)
    full_text = re.sub(r'\d+', ' ', full_text)
    words = full_text.lower().split()
    filtered_words = list(set([w for w in words if w not in stops]))
    true_articles.append(filtered_words)
    allwords.extend(filtered_words)


print(f"Processed {len(true_articles)} true news articles")

# Read in the fake training data
print()
print("Fake news articles")
fake_df = train_df[train_df['label'].str.lower() != 'true']
fake_articles = []

# Read statement + clean
for idx, row in fake_df.iterrows():
    if idx == 2000:
          break
    full_text = row['statement']
    full_text = re.sub(r'[^\w\s]', ' ', full_text)
    full_text = re.sub(r'\d+', ' ', full_text)
    words = full_text.lower().split()
    filtered_words = list(set([w for w in words if w not in stops]))
    fake_articles.append(filtered_words)
    allwords.extend(filtered_words)


print(f"Processed {len(fake_articles)} fake news articles")

# Get the 1000 most frequent words
wfreq = FreqDist(allwords)
top1000 = wfreq.most_common(1000)

training = []
traininglabel = []

# Process true articles for training
print()
print("Feature vector")
for article in true_articles:
    vec = []
    for t in top1000:
        if t[0] in article:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(1)  # 1 for true news

# Process fake articles for training
for article in fake_articles:
    vec = []
    for t in top1000:
        if t[0] in article:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append(0)  # 0 for fake news

print(f"Total training examples: {len(traininglabel)}")
print(f"Feature vector length: {len(training[0])}")

# Read testing data
testing = []
testinglabel = []

test_df = pd.read_csv("/content/NLP-Fake-News/test.tsv", sep='\t', header=None)
test_df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                  'state', 'party', 'barely_true', 'false', 'half_true',
                  'mostly_true', 'pants_on_fire', 'context']

# Print test set stats
test_true_count = len(test_df[test_df['label'].str.lower() == 'true'])
test_fake_count = len(test_df[test_df['label'].str.lower() != 'true'])
print(f"\nTest dataset: {test_true_count} true articles, {test_fake_count} fake articles")

for idx, row in test_df.iterrows():
    full_text = row['statement']
    full_text = re.sub(r'[^\w\s]', ' ', full_text)
    full_text = re.sub(r'\d+', ' ', full_text)
    words = full_text.lower().split()
    filtered_words = list(set([w for w in words if w not in stops]))

    vec = []
    for t in top1000:
        if t[0] in filtered_words:
            vec.append(1)
        else:
            vec.append(0)
    testing.append(vec)

    # Convert label to binary (true=1, false=0)
    if row['label'].lower() == 'true':
        testinglabel.append(1)
    else:
        testinglabel.append(0)

print(f"Total test examples: {len(testinglabel)}")
print(f"Feature vector length: {len(testing[0])}")

#format data to be in a specific shape for CNN
X_train = np.array(training)
X_test = np.array(testing)
y_train = np.array(traininglabel)
y_test = np.array(testinglabel)

X_train = X_train.reshape(-1, 1000, 1)
X_test = X_test.reshape(-1, 1000, 1)

print('Shape of training data:')
print(X_train.shape)
print(y_train.shape)
print('Shape of test data:')
print(X_test.shape)
print(y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten

# Initialize the model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(1000, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')  # Binary classification (true/fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate metrics
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print()
print("Test Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print()
plt.plot(range(0,15), history.history['accuracy'], color = 'blue'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy of a CNN over 15 Epochs'); plt.show()
plt.plot(range(0,15), history.history['loss'], color = 'red'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss of a CNN over 15 Epochs'); plt.show()
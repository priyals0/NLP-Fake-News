import re
import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import urllib.request

# load data
column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

train_df = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)

# preprocess
stoplist = set(stopwords.words("english"))
statement_texts = []     # original strings
statement_tokens = []    # tokenized strings

for line in train_df['statement']:
    line = str(line).strip()
    statement_texts.append(line)
    line = re.sub(r"(^| )[0-9]+($| )", r" ", line)
    tokens = [t.lower() for t in line.split() if t.lower() not in stoplist]
    statement_tokens.append(tokens)

# code to download google word2vec model - too big to push to github
'''
url = "https://github.com/eyaler/word2vec-slim/raw/refs/heads/master/GoogleNews-vectors-negative300-SLIM.bin.gz"
output_path = "GoogleNews-vectors-negative300-SLIM.bin.gz"

urllib.request.urlretrieve(url, output_path)
print("Download complete!")
'''

# load Word2Vec
# uncomment line below, couldn't push to github with it
print("Loading Word2Vec model...")
# output_path = "GoogleNews-vectors-negative300-SLIM.bin.gz"
# bigmodel = gensim.models.KeyedVectors.load_word2vec_format(output_path, binary=True)
print("Model loaded!")

# vectorize statements
statement_vectors = []
for tokens in statement_tokens:
    totvec = np.zeros(300)
    count = 0
    for w in tokens:
        if w in bigmodel:
            totvec += bigmodel[w]
            count += 1
    if count > 0:
        totvec /= count
    statement_vectors.append(totvec)

# KMeans Clustering
print("Clustering...")
k = 50  # num clusters
kmstatements = KMeans(n_clusters=k, random_state=0)
statement_clusters = kmstatements.fit_predict(statement_vectors)

# add to df
train_df['cluster'] = statement_clusters

# viewing statements in specific clusters
target_cluster = 4
print(f"Cluster {target_cluster}:")
for i in range(len(train_df)):
    if train_df.loc[i, 'cluster'] == target_cluster:
        print(train_df.loc[i, 'statement'])

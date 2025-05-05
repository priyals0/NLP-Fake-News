# NLP-Fake-News

DATASET: Fake news detection using Liar dataset (https://github.com/tfs4/liar_dataset)
- two files we used: test.tsv and train.tsv, where you can see our original testing and training data

method_baseline --> folder containing Jupyter notebook (baselines.ipynb) where we preprocessed data and ran Naive Bayes, Logistic Regression, and SVM

method_cnn --> folder containing Jupyter notebook (cnn-bagOfWords-colab.ipynb) where we preprocessed for CNN and ran CNN

method_mistral --> folder containing python file (mistral.py) where we ran Mistral, plus two text files that have Mistral's full output and it's predictions. Also contains a json for our (blank) Mistral key.
- in our code, we imported mistral from the mistralai library (https://docs.mistral.ai/getting-started/clients/) 

kmeans_clustering.py --> python file in which we ran k-means clustering as an initial data exploration

bert.py --> python file in which we tried to run bert

naivebayes_baseline.py --> python file with a Bag of Words naive bayes approach to classify news as real or fake (we ended up using this same code in baselines.ipynb)

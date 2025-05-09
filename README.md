# NLP-Fake-News

### Directory Structure, Code, Files, Running Files

The directory structure is very simple and easy to follow. Our training and test datasets are in two _.tsv_ files, _train.tsv_ and _test.tsv_. In each of our _.py_ files where we implement our methods, we load and preprocess the datasets, so to run any of the methods, just run that _.py_ file. _kmeans_clustering.py_ contains our K-Means Clustering that we performed as a part of data exploration, _baselines.py_ contains our random and majority class baselines, _BoW.py_ contains our three Bag-of-Words methods (Naive Bayes, Logistic Regression, and SVM), and inside of each of the _method_ folders, a _.ipnyb_ file exists containing the respective method. In _method-mistral_ folder, there are an additional two text files that have Mistral's full output and it's predictions, as well as a json file for our (blank) Mistral key.

### Dataset

Fake news detection using Liar dataset (https://github.com/tfs4/liar_dataset). We used two files: test.tsv and train.tsv, where you can see our original testing and training data.

### Non-Standard Libraries

We imported mistral from the mistralai library (https://docs.mistral.ai/getting-started/clients/).

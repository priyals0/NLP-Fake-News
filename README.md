# NLP-Fake-News

Fake news detection using Liar dataset (https://github.com/tfs4/liar_dataset)

naivebayes_baseline.py --> Bag of Words naive bayes approach to classify news as real or fake
news marked as true is real, all others (fake, half-true, pants-on-fire) are marked as false
<details>
<summary>📋 <strong>Classification Report</strong></summary>

<pre>
              precision    recall  f1-score   support

           0       0.84      0.95      0.89      1059
           1       0.27      0.10      0.15       208

    accuracy                           0.81      1267
   macro avg       0.55      0.52      0.52      1267
weighted avg       0.75      0.81      0.77      1267
</pre>

</details>

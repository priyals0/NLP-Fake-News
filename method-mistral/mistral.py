#note: make sure that the "mistralai" package is installed in your system

#imports:
import json
from mistralai import Mistral
import csv
import pandas as pd

#create client using your api key
f = open('apiKeys.json')
data = json.load(f)
key = data["mistral-key"]
client = Mistral(api_key=key)
#set model to be used
MODEL = "mistral-large-latest"


#create label to classification relations
true_labels = ['true', 'mostly true']
false_labels = ['false', 'mostly false', 'pants-fire']

#create map for each label and it's class (0/1)
label_mapping = {label: 1 for label in true_labels}
label_mapping.update({label: 0 for label in false_labels})


####
#Classify data using mistral prompt 
####

#need to manually add column names
column_names = ["label", "title"]
test_df = pd.read_csv("../test.tsv", sep="\t", header=None, names=column_names, usecols=[1,2], quoting=csv.QUOTE_NONE)

#filter out and only use entries labeled "true" and "false"
test_df['label'] = test_df['label'].str.strip().str.lower().map(label_mapping)
test_df = test_df.dropna(subset=['label'])

##Print useful info
print(test_df.sample(5, random_state=0))
print(f"Total Entries: {len(test_df)}")

counts = test_df['label'].value_counts()
print(f"Count of 0's: {counts.get(0, 0)}")
print(f"Count of 1's: {counts.get(1, 0)}")



#####
# Setup Prompting 
#####
import time

def runPromptOnDataFrame(promptText:str, articles_df): 
    # Fill list with response from minstral model
    predictions = []
    
    num = 1
    for index, article in articles_df.iterrows():
        headline = article["title"]
    
        # Create the prompt
        prompt = f"{promptText}\n\t{headline}"
    
        # Package prompt as a a single message that mistral will process
        MESSAGES = [{"role": "user", "content": prompt}]
    
        # This the call to Mistral with that prompt.
        completion = client.chat.complete(model=MODEL, messages=MESSAGES)
    
        # This prints the prompt:
        print(f"prompt {num}: {prompt}")
        num += 1
    
        #get prediction 
        prediction = completion.choices[0].message.content
        
        # This prints out the response
        print(f"\tPrediction: {prediction}")
        print(f"\tReal Value: {article['label']}\n\n")
        
        #add results into list
        predictions.append(prediction)
    
        #pause for a short while
        time.sleep(4)

    return predictions


#Test all data 1: Zero Shot Classification
prompt = "Is the following news headline for a real or fake news story? Answer only with '1' (real headline) or '0' (fake headline):"
#rawPredictions = runPromptOnDataFrame(prompt, test_df)  #can uncomment and rerun mistral with prompt


######
# Fetch cached prediction results (note: full detailed info of all prompts in mistral-full-output-1.txt) 
######

#fetch predictions and save as list from saved text file (since recalling minstral would take too long)
def extract_predictions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    predictions = content.split('Prediction: ')[1:]
    predictions = [p.strip() for p in predictions]
    return predictions
rawPredictions = extract_predictions('minstral-predictions-1.txt')


######
# Manually fix mistral output's that didn't stick to prompt 
######
def manuallyFindMistakes(predictions): 
    badIndexes = []
    for i, prediction in enumerate(predictions):
        if len(prediction) > 1:
            print(f"{i}: {prediction}")
            print("~"*80, "\n")
            badIndexes.append(i)
    return badIndexes

#list of indexes of bad classifications (not just 0 or 1)
mistakeIndexes = [83, 121, 228, 241, 263, 284, 286, 300, 344, 366, 428, 430, 477, 515]
corrections =    [1 , 0  ,0   , 1  , 0  ,   0,   0,   1,   0,   1,   1,   0,   0, 1]    #found out only after running manuallyFindMistakes() funct.

#find out what the prediction was trying to say
badIndexes = manuallyFindMistakes(rawPredictions)
print(badIndexes)


######
# Sanitize and isolate both lists
######

#create new sanitized predictions list (also ints only now)
y_pred = rawPredictions.copy()

for i, mistakeIndex in enumerate(mistakeIndexes):
    y_pred[mistakeIndex] = corrections[i]
y_pred = [int(prediction) for prediction in y_pred]

#create list of correct labels 
y_test = []
for index, article in test_df.iterrows(): 
    y_test.append(article['label'])



#######
# Get spefifics on wrongly classified predictions
#######
realMisclassified = []
fakeMisclassified = []
for i, pred in enumerate(y_pred):
    realVal = y_test[i]
    if pred != realVal:
        title = test_df.iloc[i]['title']
        if realVal == 1: 
            realMisclassified.append(title)
        elif realVal == 0: 
            fakeMisclassified.append(title)

for i, realMis in enumerate(fakeMisclassified): 
    print(f"{i} {realMis}")


#######
# Print Evaluation Metrics for method
#######
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def displayeConfusionMatrix(nb_cm, fmt='d'): 
    plt.figure(figsize=(5, 4))
    sns.heatmap(nb_cm, annot=True, fmt=fmt, cmap='Blues',  yticklabels=["Predicted Fake", "Predicted Real"],  xticklabels=["Actual Fake", "Actual Real"])
    plt.title("Mistral Prompting Confusion Matrix")
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()
    plt.tight_layout()
    plt.show(block=False)

#Metrics
accuracy = accuracy_score(y_test, y_pred)
print("\nMinstral:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
nb_cm = confusion_matrix(y_test, y_pred)
displayeConfusionMatrix(nb_cm)
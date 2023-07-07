# SMS_Spam_Classification
# Importing lib/Packages
import numpy as np
import pandas as pd
#Import dataset
dataset = pd.read_csv("SMSSpamCollection",sep = '\t', names=['label', 'message'])
dataset
dataset.info()
dataset.describe()
dataset['label'] = dataset['label'].map({'ham' : 0,'spam': 1})
dataset
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Countplot for spam vs ham as imbalanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data=dataset)
p = plt.title('Count for spam vs Ham as imbalance dataset')
p = plt.xlabel('is the SMS Spam')
p = plt.ylabel('Count')
#Handling imbalanced dataset using Oversumpting
only_spam = dataset[dataset["label"] == 1]
only_spam
print('No of Spam SMS :', len(only_spam))
print('No of Ham SMS :', len(dataset) - len(only_spam))
count = int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
count
for i in range(0, count-1):
    dataset = pd.concat([dataset, only_spam])
    
dataset.shape
# Countplot for spam vs ham as imbalanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data=dataset)
p = plt.title('Count for spam vs Ham as balance dataset')
p = plt.xlabel('is the SMS Spam')
p = plt.ylabel('Count')
#Creating new feature word_count
dataset['word_count'] = dataset['message'].apply(lambda x: len(x.split()))
dataset
plt.figure(figsize=(12,6))

# (1,1)
plt.subplot(1,2,1)
g = sns.histplot(dataset[dataset["label"] == 0].word_count,kde = True)
p = plt.title('Distribution of word_count for Ham SMS')

#(1,2)
plt.subplot(1,2,2)
g = sns.histplot(dataset[dataset["label"] == 1].word_count, color = "red" ,kde = True)
p = plt.title('Distribution of word_count for Spam SMS')

plt.tight_layout()
plt.show()
#Creating new feature of containing currency symbols
def currency(data):
    currency_symbols = ['€', ' $', '¥', '£', '₹']
    for i in currency_symbols:
        if i in data:
            return 1
    return 0
dataset["contains_currency_symbols"] = dataset["message"].apply(currency)
dataset
# Countplot for contains_currency_symbols
plt.figure(figsize=(8,8))
g = sns.countplot(x = 'contains_currency_symbols', data = dataset, hue = "label")
p = plt.title('Countplot for containing currency symbol')
p = plt.xlabel('Does SMS contains any currrency symbols?')
p = plt.ylabel('Count')
p = plt.legend(labels=["Ham","Spam"], loc = 9)
def number(data):
    for i in data:
        if ord(i) >= 48 and ord(i) <= 57:
            return 1
    return 0
dataset["contains_number"] = dataset["message"].apply(number)
dataset
# Countplot for containing numbers
plt.figure(figsize=(8,8))
g = sns.countplot(x = 'contains_number', data = dataset, hue = "label")
p = plt.title('Countplot for containing numbers')
p = plt.xlabel('Does SMS contains any number?')
p = plt.ylabel('Count')
p = plt.legend(labels=["Ham","Spam"], loc = 9)
# Data cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus = []

wnl = WordNetLemmatizer()
for sns in list(dataset.message):
    message = re.sub(pattern = '[^a-zA-Z]', repl='', string=sns) # Filtering out specialcharacters and numbers
    message = message.lower()
    words = message.split() #Tokenizer
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemm_words)
    
    corpus.append(message)
corpus
# creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()
X = pd.DataFrame(vectors,columns = feature_names)
y = dataset['label']
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.33, random_state=42)
X_test
# naive Bayes modal
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
cv=cross_val_score(mnb,X,y,scoring='f1',cv=10)
print(round(cv.mean(),3))
print(round(cv.std(),3))
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm
import seaborn as sns
plt.figure(figsize=(8,8))
g = sns.heatmap(data = cm, annot = True)
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Values')
p = plt.ylabel('Predicted Values')
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1 = cross_val_score(dt, X, y, scoring = 'f1', cv = 10)
print(round(cv1.mean(),3))
print(round(cv1.std(), 3))
dt.fit(X_train, y_train)
y_pred1 = dt.predict(X_test)
print(classification_report(y_test, y_pred))
cm1 = confusion_matrix(y_test, y_pred1)
cm1
plt.figure(figsize=(8,8))
g = sns.heatmap(data = cm1, annot = True)
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Values')
p = plt.ylabel('Predicted Values')
def predict_spam(sns):
    message = re.sub(pattern = '[^a-zA-Z]', repl='', string=sns) # Filtering out specialcharacters and numbers
    message = message.lower()
    words = message.split() #Tokenizer
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemm_words)
    temp = tfidf.transform([message]).toarray()
    return mnb.predict(temp)
# Prediction 1 - Lottery text message
sample_message = 'IMPORTANT - You could be entitled up to 23,160 in cpmpensation from mis-sold PPI on a credit or loan.'
if predict_spam(sample_message):
    print('Gotcha! This is a SPAM message')
else:
    print('This is a HAM (normal) message')
# Prediction 2 - Casual text message
sample_message = 'Came to think of it. I have got a spam message before'
if predict_spam(sample_message):
    print('Gotcha! This is a SPAM message')
else:
    print('This is a HAM (normal) message')


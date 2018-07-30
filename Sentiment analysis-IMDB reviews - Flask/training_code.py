#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:18:50 2018

@author: abhiyush
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


%matplotlib inline

datasets = pd.read_csv("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv", delimiter = '\t', quoting = 3, header = None)
datasets.columns = ['Reviews', 'Likes']

datasets

# =============================================================================
# #Object initialization for stemmer
# ps = PorterStemmer()
# 
# corpus = []
# 
# #Cleaning the text with stemmer
# 
# for i in range(0,1000):
#     review = re.sub('[^a-zA-Z]', ' ', datasets['Reviews'][i])
#     review = review.lower()
#     review = review.split()
#     review = [word for word in review if not word in set(stopwords.words('english'))]
#     review = [ps.stem(word) for word in review]
#     review = ' '.join(review)
#     corpus.append(review)
#     
# 
# =============================================================================
#object initialization for lemmatization
    
lemmatizer = WordNetLemmatizer()

corpus = []

#cleaning the text with lemmatizer

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', datasets['Reviews'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
    
corpus
#Creating a dataframe with column name as review
df = pd.DataFrame({'Reviews' : corpus})
df

#using count vectorizer

cv = CountVectorizer(max_features = 2300, ngram_range = (1,2))
fit_corpus = cv.fit(corpus)
print(cv.get_feature_names())

#we can directly use fit_transform 
# X = cv.fit_transform(corpus).toarray()
transform_corpus = cv.transform(corpus)

transform_corpus_toarray = transform_corpus.toarray()
print(transform_corpus_toarray)
print(transform_corpus_toarray.shape)

X = transform_corpus_toarray

save_count_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/count_vectorizer_spyder.pickle", 'wb')
pickle.dump(cv, save_count_vectorizer)
save_count_vectorizer.close()

y = datasets.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# #Naive bayes using Gaussian 
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# 
# y_pred_gaussian = classifier.predict(X_test)
# 
# cm = confusion_matrix(y_test, y_pred_gaussian)
# print("The confusion matrix is: ", cm)
# accuracy = (69+66)/200 *100
# print("Accuracy of model using Gaussian naive: ", accuracy)
# 
# =============================================================================

#Naive bayes using Multinomial 
classifier = MultinomialNB(alpha = 0.1)
classifier.fit(X_train, y_train)

y_pred_multinomial = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_multinomial)
print("The confusion matrix is: ", cm)
accuracy = (68+95)/200 *100
print("Accuracy of model using Gaussian naive: ", accuracy)

save_classifier = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/naive_bayes_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


#Using tf-idf approach
corpus
b = datasets.iloc[:,1].values
a_train, a_test, b_train, b_test = train_test_split(corpus, b, test_size = 0.2, random_state = 0)
type(a_train)

#fit and transform using tf-idf 
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2))
fit_tfidf_vectorizer = tfidf_vectorizer.fit(a_train)
print(tfidf_vectorizer.get_feature_names())
tfidf_vectorizer_transform = tfidf_vectorizer.transform(a_train)
tfidf_vectorizer_transform_toarray = tfidf_vectorizer_transform.toarray()

tfidf_vectorizer_transform_toarray.shape
a_train = tfidf_vectorizer_transform_toarray

#Naive bayes classifier 
classifier = MultinomialNB(alpha = 0.1)
classifier.fit(a_train, b_train)

#transforming test datasets using tf-idf approach
a_test  = tfidf_vectorizer.transform(a_test)
b_pred = classifier.predict(a_test)

cm = confusion_matrix(b_test, b_pred)
cm

accuracy = (68+94)/200 *100
print("The accuracy of the model using tdf-idf approach and Miultinomial naive bayes with lemmatizer is : ", accuracy)

save_tfidf_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/tfidf_vectorizer.pickle", "wb")
pickle.dump(tfidf_vectorizer, save_tfidf_vectorizer)
save_tfidf_vectorizer.close()

save_classifier_tfidf = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/naive_bayes_classifier_tfidf.pickle", "wb")
pickle.dump(classifier, save_classifier_tfidf)
save_classifier.close()
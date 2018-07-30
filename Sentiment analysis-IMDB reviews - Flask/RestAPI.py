#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:45:56 2018

@author: abhiyush
"""

from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

#pickle of count vectorizer

open_count_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/models/count_vectorizer_spyder.pickle", "rb")
cv = pickle.load(open_count_vectorizer)
open_count_vectorizer.close()

open_classifier = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/models/naive_bayes_classifier.pickle", "rb")
classifier = pickle.load(open_classifier)
open_classifier.close()

#pickle of tfidf vectorizer

open_tfidf_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/models/tfidf_vectorizer.pickle", "rb")
tfidf_vectorizer = pickle.load(open_tfidf_vectorizer)
open_tfidf_vectorizer.close()

open_classifier_tfidf = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/models/naive_bayes_classifier.pickle_tfidf", "rb")
classifier_tfidf = pickle.load(open_classifier_tfidf)
open_classifier_tfidf.close()


datasets = pd.read_csv("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/imdb_reviews.csv", delimiter = '\t', quoting = 3, header = None)
datasets.columns = ['Reviews', 'Likes']
datasets


@app.route('/', methods = ['GET', 'POST'])
def index():
    value = request.json['key']
    return value
    
@app.route('/predict', methods = ['GET','POST'])
def predict():
    datasets = request.json['Reviews']
    #reviews = datasets['Reviews']
    #datasets = reviews
    #print(reviews)

    reviews = [datasets]
    features = cv.transform(reviews).toarray()

    predictions = classifier.predict(features)
    #print(predictions)
    #return predictions
    return jsonify({'predictions' : predictions.tolist()})
  
if __name__ == '__main__':
    app.run(debug = True)
    


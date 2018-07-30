#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:15:42 2018

@author: abhiyush
"""
import pickle

#pickle of count vectorizer

open_count_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/count_vectorizer_spyder.pickle", "rb")
cv = pickle.load(open_count_vectorizer)
open_count_vectorizer.close()

open_classifier = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/naive_bayes_classifier.pickle", "rb")
classifier = pickle.load(open_classifier)
open_classifier.close()

#pickle of tfidf vectorizer

open_tfidf_vectorizer = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/tfidf_vectorizer.pickle", "rb")
tfidf_vectorizer = pickle.load(open_tfidf_vectorizer)
open_tfidf_vectorizer.close()

open_classifier_tfidf = open("/home/abhiyush/mPercept/Natural Language Processing/Sentiment Analysis/IMDB reviews/spyder/naive_bayes_classifier.pickle_tfidf", "rb")
classifier_tfidf = pickle.load(open_classifier_tfidf)
open_classifier_tfidf.close()

#Using count vectorizer

doc_pos = ["Movie is good"]
doc_pos = cv.transform(doc_pos).toarray()
classifier.predict(doc_pos)

doc_neg = ['Movie is bad']
doc_neg = cv.transform(doc_neg).toarray()
classifier.predict(doc_neg)

#Using tfidf vectorizer

doc_pos1 = ["Moive is so good"]
doc_pos1 = tfidf_vectorizer.transform(doc_pos1).toarray()
classifier_tfidf.predict(doc_pos1)

doc_neg1 = ['Movie is so bad']
doc_neg1 = tfidf_vectorizer.transform(doc_neg1).toarray()
classifier_tfidf.predict(doc_neg1)

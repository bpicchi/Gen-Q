from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser

import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer
from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB

from csv import DictReader

stop_words = set(stopwords.words('english'))


def read_csv_col(column: str) -> list:
	c = []

	with(open("data_file_br_50k.csv")) as file:
		c = [row[column] for row in DictReader(file)]

	return c

# lemmatize each word and take out stop words


def lemmatize(words: list) -> list:
	lemmatized = []
	punctuation = ['.', ',', '?', '!', ';', ':', '-', "n't"]
	stops = stopwords.words('english')

	lemmatizer = WordNetLemmatizer()

	for word in words:
		if word not in stops and word not in punctuation:
			lemmatized.append(lemmatizer.lemmatize(word.lower()))

	return lemmatized


def extract_keys(words: list) -> list:
	no_stops = []
	special_chars = [',', '"', "'", '.', '!',
		'?', ';', ':', '-', '_', '*', '$', '~', '`']

	for w in words:
		if w not in stop_words and w not in special_chars:
			no_stops.append(w)

	return no_stops


def cleanup(sentence: str) -> list:
	# tokenize sentence
	words = word_tokenize(sentence)

	# lemmatize words
	lemm_words = lemmatize(words)

	# remove stop words
	cleaned_words = extract_keys(lemm_words)

	return cleaned_words


def simple_tag(words: list) -> list:
	tagged = pos_tag(words)

	# tagged structure: (word, tag)
	# ex. ('I', 'PRP') and ('like', 'VBP')

	simple_tags = []

	for tag in tagged:
		word = tag[0]
		pos = tag[1]

		if pos[0] == 'V' or pos[0] == 'N' and not pos == 'NNP':
			simple_tags.append((word, pos[0].lower()))
		else:
			simple_tags.append((word, 'NA'))

	return simple_tags


def get_data() -> list:
    quotes = read_csv_col('quote')

	return quotes


def merge_tokens(words: list) -> str:
	merged = ""
	for index, word in enumerate(words):
		if index == len(words)-1:
			merged += word.lower()
		else:
			merged += word.lower() + " "

	return merged


def main():
    sentence = 'I am going to the river bank where I will swim and look at fish'
    tokens = cleanup(sentence)


quotes = get_data()
	for i in range(1000):
        print(quotes[i])

    print(tokens)

    sbert_model = SentenceTransformer('stsb-roberta-large')


if __name__ == "__main__":
	main()

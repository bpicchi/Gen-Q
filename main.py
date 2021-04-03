# nltk imports
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser

# file management/data imports
import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# sentence_transformer import
from sentence_transformers import SentenceTransformer


# reads csv data file and returns list of all quotes
def get_quotes() -> list:
	df = pd.read_csv('data_file_br_50k.csv', encoding='cp1252')
	return df['quote']


def lemmatize(words: list) -> list:
	lemmatized = []
	punctuation = ['.', ',', '?', '!', ';', ':', '-', "n't", '_', '*', ')', '(', '$', '~', '`', '@', '#', '%', '^', '&', ']', '[', '{', '}']
	stops = stopwords.words('english')
	lemmatizer = WordNetLemmatizer()

	for word in words:
		if word not in stops and word not in punctuation:
			lemmatized.append(lemmatizer.lemmatize(word.lower()))
	
	return lemmatized


def cleanup(sentence: str) -> list:
	# tokenize sentence
	words = word_tokenize(sentence)

	# lemmatize
	cleaned_words = lemmatize(words)

	return cleaned_words


def get_data(column: str) -> list:
	df = pd.read_csv('data_file_br_50k.csv', encoding='cp1252')

	return df[column]

def cosine(u, v):
	return np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))

def binary_relevance(quotes: list):
	print('Loading model...')

	model = SentenceTransformer('stsb-distilbert-base')

	print('Model loaded.')
	print('Encoding database of quotes...')

	sentence_embeddings = model.encode(quotes)
	
	print('Encoded quotes.')

	query = "My family left me and I am sad"
	query_vec = model.encode([query])[0]

	# print similarity between input (query) and each quote in data
	# for q in quotes:
	# 	sim = cosine(query_vec, model.encode([q])[0])
	# 	print("Quote = ", q, "\tSimilarity = ", sim)


def main():
	print('Retrieving quotes...')

	quotes = get_data('quote')
	
	print('Retrieved quotes.')

	binary_relevance(quotes)


if __name__ == "__main__":
	main()
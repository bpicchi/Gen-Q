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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# sentence_transformer import
from sentence_transformers import SentenceTransformer

# matplotlib import
from matplotlib import pyplot as plt


# reads csv data file and returns list of all quotes
def get_quotes() -> list:
    df = pd.read_csv('data_file.csv', encoding='cp1252')
    return df['quote']


def lemmatize(words: list) -> list:
    lemmatized = []
    punctuation = ['.', ',', '?', '!', ';', ':', '-', "n't", '_', '*', ')', '(', '$', '~', '`', '@', '#', '%', '^', '&',
                   ']', '[', '{', '}']
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
    df = pd.read_csv('data_file.csv', encoding='cp1252')

    return df[column]


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def bert(quotes: list):
    print('Loading model...')

    model = SentenceTransformer('stsb-distilbert-base')

    print('Model loaded.')
    print('Encoding database of quotes...')
    sentence_embeddings = []
    for i in range(200):
        sentence_embeddings.append(model.encode(quotes[i]))

    print('Encoded quotes.')

    query = "My family left me and I am sad"
    query_vec = model.encode([query])[0]
    print(query_vec)

def bert2(quotes: list):
    print('Loading model...')
    model = SentenceTransformer('stsb-distilbert-base')
    print('Model loaded.')
    print('Encoding database of quotes...')

    quotes_matrix = []
    quotes2 = []
	# HERES WHERE YOU SPECIFY THE NUMBER OF QUOTES YOU WANT TO WORK WITH
    for i in range(1000):
        quotes2.append(quotes[i])
    for q in quotes2:
        print(q)
        encoding = model.encode([q])
        quotes_matrix.append(encoding[0])
    X = np.array(quotes_matrix)
    model = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100, random_state=1)
    predict = model.fit_predict(X)
    i = 0
    for ind, q in enumerate(quotes2):
        if i == 30: break
        print(str(predict[ind]) + ": " + str(q))
        i += 1

	# ELBOW METHOD PLOT TO FIND BEST # OF CLUSTERS
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


# print similarity between input (query) and each quote in data
# for q in quotes:
# 	sim = cosine(query_vec, model.encode([q])[0])
# 	print("Quote = ", q, "\tSimilarity = ", sim)

def k_means(quotes: list):
    model = SentenceTransformer('stsb-distilbert-base')
    sentence_embeddings = model.encode(quotes)


    # for i in range(200):
    # Using the elbow method to determine the optimal # of clusters
    # wcss = []
    # for i in range(1, 11):
    # 	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # 	kmeans.fit(X)
    # 	wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    # once the ideal cluster is found, plug in for 4 below and run code to see results

    # kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # pred_y = kmeans.fit_predict(quotes)
    # plt.scatter(X[:,0], X[:,1])
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    # plt.show()



def main():
    print('Retrieving quotes...')
    quotes = get_data('quote')

    # k_means(quotes)


    print('Retrieved quotes.')

    bert2(quotes)


if __name__ == "__main__":
    main()
    # model = SentenceTransformer('stsb-distilbert-base')
    # sentences = ['This framework generates embeddings for each input sentence',
    # 			 'Sentences are passed as a list of string.',
    # 			 'The quick brown fox jumps over the lazy dog.']
    #
    # # Sentences are encoded by calling model.encode()
    # embeddings = model.encode(sentences)
    #
    # # Print the embeddings
    # for sentence, embedding in zip(sentences, embeddings):
    # 	print("Sentence:", sentence)
    # 	print("Embedding:", embedding)
    # 	print("")

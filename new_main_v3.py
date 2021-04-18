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
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

# sentence_transformer import
from sentence_transformers import SentenceTransformer

# matplotlib import
from matplotlib import pyplot as plt

# pickle import
import pickle

# ssl import
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk

nltk.download('stopwords')


def tokenize(inpQuotes):
    sent = inpQuotes.split()
    myStart = ['"', "'"]
    myEnd = ["n't", '.', ',', '"', "'", '?', ';', '!', ':', "'re", "'ll", "'m", "'s"]
    newTokens = []

    def aux(token):
        token = token.lower()
        if token == 'wo':
            newTokens.append('will')
            return
        start = next((t for t in myStart if token.startswith(t)), None)
        if start:
            n = len(start)
            if token[n:] == 'cause':
                aux('because')
            else:
                aux(token[n:])
            return
        end = next((t for t in myEnd if token.endswith(t)), None)
        if end:
            n = len(end)
            t1, t2 = token[:-n], token[-n:]
            aux(t1)
            if t2 == "n't":
                newTokens.append('not')
            elif t2 == "'re":
                newTokens.append('are')
            elif t2 == "'ll":
                newTokens.append('will')
            elif t2 == "'m":
                newTokens.append('am')
            elif t2 == "'s" and (t1 == 'it' or t1 == 'there' or t1 == 'he' or t1 == 'she'
                                 or t1 == 'that'):
                newTokens.append('is')
            return
        if not (token == '-' or token == '--'): newTokens.append(token)

    for s in sent: aux(s)
    return newTokens

def lemmatizeQuotes(inpQuotes: list) -> list:
    toRet = []
    stops = stopwords.words('english')
    for q in inpQuotes:
        if q not in stops: toRet.append(WordNetLemmatizer().lemmatize(q))
    return toRet

def pklStoreData(finalQuotes: bytearray, fname: str):
    newArray = finalQuotes.astype('i')
    dbfile = open(fname, 'ab')
    pickle.dump(newArray, dbfile)
    dbfile.close()

def pklLoadData(fname: str):
    dbfile = open(fname, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()

    # for keys in db:
    #     print(keys, '=>', db[keys])
    # dbfile.close()

# Diedra's KMeans
def kmeans_function(quotes, model, inp) -> bytearray:
    quotes_matrix = []
    quotes2 = []
    quotesToPrint = []
    for i in range(140000):
        quotesToPrint.append(quotes[i])
    quotesToPrint.append(inp)
    for i in range(140000):
        tokenized = tokenize(quotes[i])
        lemmatized = lemmatizeQuotes(tokenized)
        print("lem: ", lemmatized)
        quotes2.append(lemmatized)
    inpTok = tokenize(inp)
    inpLem = lemmatizeQuotes(inpTok)
    print("lem: ", inpLem)
    quotes2.append(inpLem)
    for q in quotes2:
        # issue with zero index and axis 0 occurs here
        if not q: continue
        encoding = model.encode(q)
        quotes_matrix.append(encoding[0])
    #   send quotes to array
    X = np.array(quotes_matrix)
    # call kmeans and choose best silhouette score
    myRange = range(2, 6)
    optNumClusters = 2
    bestSilhouette = -1
    for i in myRange:
        print("trying range ", i)
        kmodel = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=100, random_state=1)
        predict = kmodel.fit_predict(X)
        klabels = kmodel.labels_
        silhouette_score = metrics.silhouette_score(X, klabels, metric='euclidean')
        if silhouette_score > bestSilhouette and silhouette_score > 0:
            bestSilhouette = silhouette_score
            optNumClusters = i
    # get best number of clusters
    kmodel = KMeans(n_clusters=optNumClusters, init='k-means++', max_iter=200, n_init=100, random_state=1)
    predict = kmodel.fit_predict(X)
    print('**** Silhouette Analysis ****')
    print('Silhouette Score: ', bestSilhouette)
    # get prediction of input
    print("**** Now predictions ******")
    inputClust = predict[140000]
    print("Input prediction: " + str(inputClust) + " -- Input: " + str(quotes2[30]))
    i = 0
    for ind, q in enumerate(quotesToPrint):
        if i == 140000: break
        if predict[ind] == inputClust: print(str(predict[ind]) + ": " + str(q))
        i += 1
    return X

# TODO: Blaise -- messing around with this method
def synonym_intersection(quotes2: list, inp):
    length = len(quotes2) + 1
    inpTok = tokenize(inp)
    inpLem = lemmatizeQuotes(inpTok)
    print("lem: ", inpLem)
    quotes2.append(inpLem)
    for q in quotes2:
        # issue with zero index and axis 0 occurs here
        try:
            encoding = model.encode(q)
            quotes_matrix.append(encoding[0])
        except:
            print()
    #   send quotes to array
    X = np.array(quotes_matrix)
    # call kmeans and choose best silhouette score
    myRange = range(2, 6)
    optNumClusters = 2
    bestSilhouette = -1
    for i in myRange:
        print("trying range ", i)
        kmodel = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=100, random_state=1)
        predict = kmodel.fit_predict(X)
        klabels = kmodel.labels_
        silhouette_score = metrics.silhouette_score(X, klabels, metric='euclidean')
        if silhouette_score > bestSilhouette and silhouette_score > 0:
            bestSilhouette = silhouette_score
            optNumClusters = i
    # get best number of clusters
    kmodel = KMeans(n_clusters=optNumClusters, init='k-means++', max_iter=200, n_init=100, random_state=1)
    predict = kmodel.fit_predict(X)
    print('**** Silhouette Analysis ****')
    print('Silhouette Score: ', bestSilhouette)
    # get prediction of input
    print("**** Now predictions ******")
    inputClust = predict[length]
    print("Input prediction: " + str(inputClust) + " -- Input: " + str(quotes2[length]))
    i = 0
    for ind, q in enumerate(quotesToPrint):
        if predict[ind] == inputClust: print(str(predict[ind]) + ": " + str(q))
        i += 1

# Blake's synonym finder
# returns a list of pos tags
def tag(words: list) -> list:
    tagged = pos_tag(words)

    simple_tags = []

    for tag in tagged:
        word = tag[0]
        pos = tag[1]

        if pos[0] == 'V' or pos[0] == 'N' or pos[0] == 'J' and not pos == 'NNP':
            if pos[0] == 'J':
                simple_tags.append((word, 'a'))
            else:
                simple_tags.append((word, pos[0].lower()))
        else:
            simple_tags.append((word, 'na'))

    return simple_tags

# returns a list of combinations of all words and their synonyms
def find_combinations(cleaned_in: list, synonyms: dict) -> list:
    combinations = []

    def dfs(index: int, curr_sent: list):
        # when past the last word
        if index == len(cleaned_in):
            combinations.append(list(curr_sent))
            return

        word = cleaned_in[index]
        if word not in synonyms:
            # add to current running string
            curr_sent.append(word)
            # perform dfs on next word
            dfs(index + 1, curr_sent)
            curr_sent.pop(len(curr_sent) - 1)
        else:
            for syn in synonyms[word]:
                curr_sent.append(syn)
                dfs(index + 1, curr_sent)
                # pop most recently added word
                curr_sent.pop(len(curr_sent) - 1)
        return

    # outer scope fn
    dfs(0, [])
    return combinations

# returns a list of all possible combinations of sentence and its synonyms
def get_synonyms(sentence: str, words: list) -> list:
    # perform pos tagging on words
    tagged = tag(words)

    # find synonyms of each word
    synonyms = {}
    for t in tagged:
        word = t[0]
        pos = t[1]
        syn = lesk(sentence, word, pos)

        # if there is a synset, add all synonyms to corresponding word in map
        if syn:
            synonym_list = []
            for synonym in syn.lemmas():
                syn_name = synonym.name()

                # remove all '_' occurances
                new_name = ""
                for ch in syn_name:
                    if ch == '_':
                        new_name += ' '
                    else:
                        new_name += ch

                synonym_list.append(new_name)

            synonyms[word] = synonym_list

    print('synonyms: ', synonyms)

    combinations = find_combinations(words, synonyms)
    return combinations



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
    for i in range(140000):
        quotes2.append(quotes[i])
    for q in quotes2:
        encoding = model.encode([q])
        quotes_matrix.append(encoding[0])
    X = np.array(quotes_matrix)

    model = KMeans(n_clusters=4, init='k-means++', max_iter=200, n_init=100, random_state=1)
    predict = model.fit_predict(X)
    i = 0
    for idx, quote in enumerate(quotes2):
        if i == 1400000: break
        print(str(predict[idx]) + ": " + str(quote))
        i += 1


#


# print similarity between input (query) and each quote in data
# for q in quotes:
# 	sim = cosine(query_vec, model.encode([q])[0])
# 	print("Quote = ", q, "\tSimilarity = ", sim)

def main():
    print('Retrieving quotes...')
    quotes = get_data('quote')
    print('Retrieved quotes.')
    model = SentenceTransformer('stsb-distilbert-base')

    # store sentence encodings in array so we can load store it as pkl file
    X = kmeans_function(quotes, model, 'I am sad today')
    # To use:
    # 1) create blank txt file in same folder
    # 2) pass the file name (no extension) as 2nd param in storeData and 1st param in loadData
    # 3) MUST USE a different file (or delete the original pkl file) whenever you want
    # to use different sentence encodings (i.e. if you want to up the range from 1000 to 10000)
    pklStoreData(X, 'testFile')
    pklLoadData('testFile')

    # inpt = "I am sad today. I also happen to be deeply silly and terrified."
    # cleaned_in = cleanup(inpt)
    # combinations = get_synonyms(inpt, cleaned_in)
    # print()
    # for c in combinations: print(c)


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

# # ELBOW METHOD PLOT TO FIND BEST # OF CLUSTERS
#     print('Elbow method...')
#     wcss = []
#     for i in range(1, 11):
#         ah_model = AgglomerativeClustering(n_clusters=i)
#         ah_model.fit(X)
#         # kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#         # kmeans.fit(X)
#         wcss.append(kmeans.inertia_)
#     plt.plot(range(1, 11), wcss)
#     plt.title('Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('WCSS')
#     plt.show()

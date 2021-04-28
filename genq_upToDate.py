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
from csv import DictReader
import pandas as pd
import numpy as np
from numpy import savetxt
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
from sklearn import metrics

# sentence_transformer import
from sentence_transformers import SentenceTransformer

# matplotlib import
from matplotlib import pyplot as plt

# timer import
import time

# counter import
from collections import Counter


# ================ End Imports ==================== #


def get_data(column: str) -> list:
    df = pd.read_csv('data_file.csv', encoding='cp1252')

    return df[column]


# =============== Data Sanitization FNs ============ #

def lemmatize(words: list) -> list:
    lemmatized = []
    punctuation = ['.', ',', '?', '!', ';', ':', '-', "n't", '_', '*', ')', '(', '$', '~', '`', '@', '#', '%', '^', '&',
                   ']', '[', '{', '}']
    stops = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    for word in words:
        if word.lower() not in stops and word.lower() not in punctuation:
            lemmatized.append(lemmatizer.lemmatize(word.lower()))

    return lemmatized


def cleanup(sentence: str) -> list:
    # tokenize sentence
    words = word_tokenize(sentence)

    # lemmatize
    cleaned_words = lemmatize(words)

    return cleaned_words


# =============== End Sanitization FNs ============ #

# =============== Synonym FNs ===================== #

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

    combinations = find_combinations(words, synonyms)

    # convert from tokens to sentence form
    res = []
    for com in combinations:
        sent = ""

        for idx, word in enumerate(com):
            if idx == 0:
                sent += word
            else:
                sent += " " + word
        res.append(sent)

    return res


# =============== End Synonym FNs ===================== #

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def load_loc_model():
    print('Loading saved encoded quotes...')
    m = np.loadtxt(open('bert_model.csv', 'rb'), delimiter=',').astype("float")
    print('Encoded quotes loaded.\n')
    return m


def bert(num_quotes, user_input, quotes: list, m, syns) -> list:
    print('Loading stsb-distilbert-base model...')
    model = SentenceTransformer('stsb-distilbert-base')
    print('Model loaded.\n')

    query = user_input
    query_vec = model.encode([query])[0]

    results = []

    def save_bert():
        tmp_model = []
        for idx, q in enumerate(quotes):
            if idx % 1000 == 0:
                # print status update
                print(idx)
            tmp_model.append(model.encode([q])[0])

        np_model = np.array(tmp_model)
        savetxt('bert_model.csv', np_model, delimiter=',')

    print('Performing cosine similarity...')

    # calculate cosine similarity between query vector and saved model
    for idx, q in enumerate(quotes):
        sim = cosine(query_vec, m[idx])
        results.append((idx, sim))

    def sim_sorter(t):
        return t[1]

    # results are sorted from best fit to worst
    # results --> [ (quote_index, similarity) ]
    results_sorted = sorted(results, key=sim_sorter, reverse=True)

    def save_res():
        new_file = open('similarity.csv', 'w', newline='')
        csv_wirter = csv.writer(new_file)

        for r in results:
            row = [r[0], r[1]]
            csv_wirter.writerow(row)

    k_res = top_k(results_sorted)

    to_print = []

    for r in k_res:
        to_print.append(quotes[r[0]])

    print('\nSending top-k quotes to KMeans...\n')

    kmeans_function(num_quotes, k_res, m, model, syns, to_print)

    return results_sorted


def kmeans_function(num_prints, results, m, model, inpAndSyns, quotesToPrint):
    quotes_matrix = []
    quotes2 = []
    # quotesToPrint = []
    numData = 0
    for r in results:
        quote = r[0]
        encoding = m[quote]
        quotes_matrix.append(encoding)
        numData += 1
    if inpAndSyns:
        tempC = 0
        for i in inpAndSyns:
            if tempC > 50: break
            encoding = model.encode([i])
            quotes_matrix.append(encoding[0])
            tempC += 1
    #   send quotes to array
    X = np.array(quotes_matrix)
    # call kmeans and choose best silhouette score
    myRange = range(2, 6)
    optNumClusters = 2
    bestSilhouette = -1
    for i in myRange:
        print("trying range ", i)
        try:
            kmodel = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=100, random_state=1)
            predict = kmodel.fit_predict(X)
            klabels = kmodel.labels_
            silhouette_score = metrics.silhouette_score(X, klabels, metric='euclidean')
            if silhouette_score > bestSilhouette and silhouette_score > 0:
                bestSilhouette = silhouette_score
                optNumClusters = i
        except:
            continue
    # get best number of clusters
    kmodel = KMeans(n_clusters=optNumClusters, init='k-means++', max_iter=200, n_init=100, random_state=1)
    predict = kmodel.fit_predict(X)
    print('\n**** Silhouette Analysis ****')
    print('Silhouette Score: ', bestSilhouette)
    # get prediction of input
    print("\n**** Now predictions ******")
    overlaps = []
    for ind, q in enumerate(quotesToPrint):
        if i == numData: break
        overlaps.append(predict[ind])
        i += 1

    myCounter = Counter()
    for i in range(numData, len(predict)):
        inputClust = predict[i]
        if inputClust in overlaps:
            myCounter.update([inputClust])
    finalToPrint = []
    if myCounter:
        maxCluster = myCounter.most_common(1)[0][0]
        print("Counter", myCounter)
        print("max cluster", maxCluster)
        print("Input prediction: " + str(maxCluster) + " -- Input: " + str(inpAndSyns[0]))
        i = 0
        for ind, q in enumerate(quotesToPrint):
            # if i == numData: break
            if len(finalToPrint) >= num_prints: break
            if predict[ind] == maxCluster:
                if (str(predict[ind]) + ": " + str(q)) not in finalToPrint:
                    finalToPrint.append(str(predict[ind]) + ": " + str(q))
            i += 1
    else: finalToPrint = quotesToPrint
    print("\n***********")
    for p in finalToPrint: print('-  ' + p)
    print("***********")

# return a list of top 10%
def top_k(results: list) -> list:
    k_results = []

    top_sim = results[0][1]

    for r in results:
        if top_sim - r[1] > 0.1 or len(k_results) > 50:
            break
        k_results.append(r)

    return k_results


def main():
    # retrieve quotes from data_file.csv
    print('Retrieving quotes...')
    quotes = get_data('quote')
    print('Retrieved quotes.\n')

    start_t = time.perf_counter()
    m = load_loc_model()
    stop_t = time.perf_counter()
    load_t = (stop_t - start_t) / 60
    print('Time to load saved model:', load_t, '\n')

    while (True):
        user_input = input('Input: ')
        num_quotes = int(input('Num. of quotes: '))

        if user_input == '-1':
            break

        # sanitize input
        cleaned_input = cleanup(user_input)

        # find synonyms
        syns = get_synonyms(user_input, cleaned_input)
        syns.insert(0, user_input)

        # call bert and kmeans
        cs = bert(num_quotes, user_input, quotes, m, syns)
        print('==================================\n')


if __name__ == "__main__":
    main()

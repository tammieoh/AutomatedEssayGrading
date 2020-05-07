import nltk
import string
import re
import random
import csv
import codecs
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import contractions
import collections
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from io import StringIO
from csv import reader
from collections import Counter

path = "/Users/tammieoh/Desktop/training_set_rel3.tsv"
path2 = "/Users/tammieoh/Desktop/BigCorpus_5000.csv"
csv_table = pd.read_table(path,sep='\t', encoding="ISO-8859-1")

# nltk corpus: words (nltk's corpus)
word_list = words.words()
word_set = set(word_list) # turning word_list into an array

# setting up for pre-processing and tokenization
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
contraction_mapping = contractions.CONTRACTION_MAP

# create a list for the corpus after reading csv file, and lemmatize all the words
corpus_set = []
with open(path2, newline='', encoding="ISO-8859-1") as file:
    # contents = file.read()
    contents = csv.reader(file, delimiter=' ')
    for row in contents:
        lemma = wordnet_lemmatizer.lemmatize(row[0].lower())
        corpus_set.append(lemma)

# method to label data
def writer_label(points):
    if points < 8:
        return 0
    else:
        return 1

# method to find spelling errors
misspelled = []
def spelling_errors(words):
    spelling_error = 0
    misspell_arr = []
    for word in words:
        if word.isupper() is False:
            word = word.lower()
            if word not in stop_words:
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in word_set:
                    spelling_error += 1
                    misspell_arr.append(word.lower())
    misspelled.append(misspell_arr)
    return spelling_error

# method to find sophisticated words
sophist_arr = []
def sophisticated(words):
    sophisticated_words = 0
    sophist_words = []
    for word in words:
        if word.isupper() is False:
            word = word.lower()
            if word not in stop_words:
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in corpus_set:
                    sophisticated_words += 1
                    sophist_words.append(word.lower())
    sophist_arr.append(sophist_words)
    return sophisticated_words

# creating a dataframe for all features + label
keep_col = ['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1']
mod_data = csv_table[keep_col]
mod_data = mod_data.dropna(subset=['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1'], how='any')
mod_data = mod_data[mod_data.essay_set == 1]
rater1_domain_feat = np.ndarray.tolist(mod_data['rater1_domain1'].values)
rater2_domain_feat = np.ndarray.tolist(mod_data['rater2_domain1'].values)
mod_data = mod_data.assign(total_score=(mod_data['rater1_domain1'] + mod_data['rater2_domain1']))
mod_data = mod_data.assign(label = mod_data['total_score'].apply(writer_label))
essay_feats = np.ndarray.tolist(mod_data['essay'].values)
labels = np.ndarray.tolist(mod_data['label'].values)



spell_check = []
num_sophisticated = []

#  cleaning up text to check for spelling errors and sophisticated words
for element in mod_data['essay']:
    text = word_tokenize(element)
    words = [word for word in text if word.isalpha()]
    spell_check.append(spelling_errors(words))
    num_sophisticated.append(sophisticated(words))

#  checking if sophisticated words are misspelled, and if so, lower the num of sophisticated words
counter = []
for soph_words in sophist_arr:
    count = 0
    for word in soph_words:
        if word in misspelled:
            count += 1
    counter.append(count)
sophisticated_revised = []
for i in range(len(counter)):
    sophisticated_num = num_sophisticated[i] - counter[i]
    sophisticated_revised.append(sophisticated_num)

#  checking for grammar mistakes with raw essay, no pre-processing
grammar_mistakes = []
for element in mod_data['essay']:
    gram_error = 0
    indices = []
    sentence_count = []
    tagged_sentences = []
    for sentences in re.split('(?<=[.!?]) +', element):
        tagged_sentences.append(nltk.pos_tag(sentences.split()))
    count = 0
    for tagged in tagged_sentences:
        for i in range(len(tagged)):
            if i != len(tagged) - 1:
                if tagged[i][0] == "their":
                    if tagged[i+1][1] in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == "there":
                    if tagged[i + 1][1] in {'JJ', 'NN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == "they're":
                    if tagged[i + 1][1] in {'CC', 'NNS'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == "your":
                    if tagged[i+1][1] in {'CC', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == "you're":
                    if tagged[i + 1][1] in {'CC', 'NNS'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == ",":
                    if tagged[i + 1][0] in {'because', 'however'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                    elif tagged[i+1][0].isupper() == True:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                    elif tagged[i+1][1] in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == '"':
                    if tagged[i+1][0] == ',':
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][1] in {'NNS', 'NNPS'} and tagged[i][0].lower() not in contraction_mapping:
                    if '.' not in tagged[i][0] and '?' not in tagged[i][1] and '!'not in tagged[i][1]:
                        if tagged[i+1][1] == 'VBZ':
                            gram_error += 1
                            sentence_count.append(count)
                            indices.append(i)
                elif tagged[i][1] in {'NN', 'NNP'} and tagged[i][0].lower() not in contraction_mapping:
                    if '.' not in tagged[i][0] and '?' not in tagged[i][1] and '!' not in tagged[i][1]:
                        if tagged[i+1][1] in {'VB'}:
                            gram_error += 1
                            sentence_count.append(count)
                            indices.append(i)
                elif tagged[i][0].lower() in contraction_mapping:
                    gram_error += 1
                    sentence_count.append(count)
                    indices.append(i)
                # mixing up possessive and plural forms
                # sister's car vs sisters car
                elif tagged[i+1][1] in {'NN', 'NNS', 'NNP', 'NNP'}:
                    if tagged[i][1] == 'NNS':
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == 'this':
                    if tagged[i+1][1] in {'NNS', 'NNPS'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
                elif tagged[i][0] == 'these':
                    if tagged[i + 1][1] in {'NN', 'NNP'}:
                        gram_error += 1
                        sentence_count.append(count)
                        indices.append(i)
        count += 1
    grammar_mistakes.append(gram_error)

essay_length = []
sentence_length = []
avg_sentences = []
lex_div = []

#  calculating other remaining features
for element in essay_feats:
    # count num of words in the essay
    num_words = len(element.split())
    essay_length.append(num_words)
    # count the number of unique words
    unique_words = Counter(element.split())
    lex_div.append(len(unique_words))
    # count num of sentences in the essay and add it to an array
    num_sentences = textstat.sentence_count(element)
    sentence_length.append(num_sentences)
    # calculate average sentence length
    avg_sent_length = num_words/num_sentences
    avg_sentences.append(avg_sent_length)

# convert all the arrays into numpy arrays to add into dataframe
essay_numwords = np.array(essay_length)
essay_numsent = np.array(sentence_length)
essay_avgsentlen = np.array(avg_sentences)
essay_lexdiv = np.array(lex_div)
essay_spelling = np.array(spell_check)
essay_sophisticated = np.array(sophisticated_revised)
essay_grammar = np.array(grammar_mistakes)

# adding columns into dataframe
mod_data = mod_data.assign(essay_length=essay_numwords)
mod_data = mod_data.assign(num_sentences=essay_numsent)
mod_data = mod_data.assign(avg_sent_length=essay_avgsentlen)
mod_data = mod_data.assign(lexical_diversity=essay_lexdiv)
mod_data = mod_data.assign(spelling_error=essay_spelling)
mod_data = mod_data.assign(sophisticated_words=essay_sophisticated)
mod_data = mod_data.assign(grammar_error=essay_grammar)
mod_data = mod_data.assign(label=mod_data['total_score'].apply(writer_label))

# all scatter plots of the features' correlation with the label
mod_data.plot.scatter(x = 'essay_length', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'num_sentences', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'avg_sent_length', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'lexical_diversity', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'spelling_error', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'sophisticated_words', y = 'total_score', s=10)
mod_data.plot.scatter(x = 'grammar_error', y = 'total_score', s=10)

# correlation coefficients between feature and label
r = np.corrcoef(mod_data['essay_length'], mod_data['total_score'])
essay_cor = r[0,1]
r = np.corrcoef(mod_data['num_sentences'], mod_data['total_score'])
numsent_cor = r[0,1]
r = np.corrcoef(mod_data['avg_sent_length'], mod_data['total_score'])
avg_sent_cor = r[0,1]
r = np.corrcoef(mod_data['lexical_diversity'], mod_data['total_score'])
lexdiv_cor = r[0,1]
r = np.corrcoef(mod_data['spelling_error'], mod_data['total_score'])
spelling_cor = r[0,1]
r = np.corrcoef(mod_data['sophisticated_words'], mod_data['total_score'])
sophisticated_cor = r[0,1]
r = np.corrcoef(mod_data['grammar_error'], mod_data['total_score'])
grammar_cor = r[0,1]

X_feat = mod_data.iloc[:, 7:].to_numpy()
Y_label = mod_data['label'].to_numpy()
Y_label = Y_label.reshape(Y_label.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X_feat, Y_label, test_size=0.3, random_state=30)
logistic_regression = LogisticRegression(solver='newton-cg').fit(X_train, y_train)
print(classification_report(y_test, logistic_regression.predict(X_test), digits=3))
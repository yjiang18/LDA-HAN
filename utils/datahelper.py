import numpy as np
import re
import nltk
import sys
import csv
import os
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Dataset(object):

    def __init__(self):
        self.file_path = './data/articles-manual-all-sents1.tsv'
        self.maxdoclen = 0
        self.maxsentlen = 0
        self.pad_doclen = 30
        self.pad_sentlen = 40
        self.tokens = []
        self.labels = []
        self.new_stop = ['splt']
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stopwords = self.stopwords.union(self.new_stop)
        self.punctuation = string.punctuation
        self.extra_punctuation = ["\'\'", "``", "-rrb-", "-lrb-", "\'s"]

    def preprocessing(self, text):
        text = text.lower()
        text = re.sub("'", '', text)
        text = re.sub("\.{2,}", '.', text)
        text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
        text = re.sub('\.', ' . ', text)
        text = re.sub('\?', ' ? ', text)
        text = re.sub('!', ' ! ', text)

        # tokenize
        text = text.split()
        return text

    def normalize_words(self, words):
        token = self.preprocessing(words)
        cleaned_token = []
        for w in token:
            if w not in self.stopwords and w not in self.punctuation and w not in self.extra_punctuation:
                cleaned_token.append(w)
        return cleaned_token


    def sentence_word_spliter(self, text):
        sentences = []
        sentence = []
        for t in text:
            if t not in ['.', '!', '?','splt']:
                sentence.append(t)
            else:
                sentence.append(t)
                sentences.append(sentence)
                if len(sentence) > self.maxsentlen:
                    self.maxsentlen = len(sentence)
                sentence = []
        if len(sentence) > 0:
            sentences.append(sentence)

        sentences = [sent for sent in sentences if len(sent) > 1]  # remove ['!'], ['!']

        # add split sentences to tokens
        self.tokens.append(sentences)
        if len(sentences) > self.maxdoclen:
            self.maxdoclen = len(sentences)

    def data_reader(self, hierachical_data=False):
        self.rawtext = []
        y_encoder = LabelEncoder()
        with open(self.file_path, 'r') as f:
            lineno = 0
            csv_read = csv.reader(f, delimiter='\t')
            for line in csv_read:
                lineno += 1
                sys.stdout.write("processing line %i     \r" % lineno)
                sys.stdout.flush()

                text = line[4]

                label = line[1]

                cleaned_text = self.normalize_words(text)
                self.rawtext.append(cleaned_text)

                text = self.preprocessing(text)
                if hierachical_data:
                    if len(text) == 0:
                        continue
                    self.sentence_word_spliter(text)
                else:
                    self.tokens.append(text)
                self.labels.append(label)

        self.labels = y_encoder.fit_transform(self.labels)


    def load_data(self, hierachical_data=False):
        if hierachical_data:
            sentences = [sent for doc in self.tokens for sent in doc]
        else:
            sentences = [sent for sent in self.tokens]

        self.tokenizer = Tokenizer()

        print("Starting tokenizing.")
        self.tokenizer.fit_on_texts(sentences)

        word_index = self.tokenizer.word_index

        if hierachical_data:
            padded_doc = []
            for i, val in enumerate(self.tokens):
                seq = self.tokenizer.texts_to_sequences(val)
                padded_seq = pad_sequences(seq, maxlen=self.pad_sentlen)
                padded_doc.append(padded_seq)

            padded_doc = pad_sequences(padded_doc, maxlen=self.pad_doclen)

            return padded_doc, self.labels, word_index, self.tokenizer
        else:
            seq = pad_sequences(self.tokenizer.texts_to_sequences(sentences), maxlen=self.pad_sentlen)

            return seq, self.labels, word_index, self.tokenizer

    def train_val_test(self, text, doc_topics, labels):

        x = list(zip(text, doc_topics))

        X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.1, random_state=1)
        train_seq, train_doc_topics = zip(*X_train)
        test_seq, test_doc_topics = zip(*X_test)

        return train_seq, train_doc_topics, y_train, \
               test_seq, test_doc_topics, y_test



from collections import Counter
import math
import os
import pickle
import re
import string

import pysrt
import nltk
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
from sklearn.cluster import MeanShift, estimate_bandwidth

URL_REGEX = '(?:www\.)|(?:https?://)\S+\.\S+'
STOP_WORDS = set(nltk.corpus.stopwords.words('english')) | {"n't", "..."} | set(string.punctuation)
STEMMER = nltk.stem.porter.PorterStemmer()

def extract_features(subtitle_lines):
    word_counter = Counter()
    for line in subtitle_lines:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            token = token.lower()
            # Skipping stop words, tokens starting with "'" and one reoeated character (like '---' and such)
            if token in STOP_WORDS or token.startswith("'") or len(set(token)) == 1:
                continue
            # Stemming
            token = STEMMER.stem(token)
            word_counter[token] += 1
    return word_counter


def extract_features_ngrams(subtitle_lines, n=2):
    word_counter = Counter()
    for line in subtitle_lines:
        tokens = nltk.word_tokenize(line)
        for i in xrange(len(tokens) - (n - 1)):
            ngram = []
            for j in xrange(n):
                token = tokens[i+j].lower()
                # Skipping stop words, tokens starting with "'" and one reoeated character (like '---' and such)
                if token in STOP_WORDS or token.startswith("'") or len(set(token)) == 1:
                    continue
                # Stemming
                token = STEMMER.stem(token)
                ngram.append(token)
            ngram = tuple(ngram)
            word_counter[ngram] += 1
    return word_counter

extract_features = extract_features_ngrams

def extract_lines(subtitle_path):
    try:
        subtitle_object = pysrt.open(subtitle_path)
    except UnicodeDecodeError:
        subtitle_object = pysrt.open(subtitle_path, encoding='latin1')
    subtitle_lines = []
    for sub in subtitle_object:
        text = sub.text
        # Removing any formatting via HTML tags
        text = re.sub('<[^<]+?>', '', text)
        # Skipping links (usually ads or subtitle credits so irrelevant)
        if re.search(URL_REGEX, text):
            continue
        subtitle_lines.append(text)
    return subtitle_lines

def classification_validation(features, labels):
    kf = sklearn.cross_validation.StratifiedKFold(labels, n_folds=3)
    clf_scores = []
    print '# CROSS-VALIDATION'
    for train, test in kf:
        X_train, X_test, y_train, y_test = features[train], features[test], labels[train], labels[test]
        clf = get_classifier().fit(X_train, y_train)

        clf_score = clf.score(X_test, y_test)
        clf_scores.append(clf_score)
        print "  . Score: {}".format(clf_score)

    avg_clf_score = sum(clf_scores) / len(clf_scores)
    print
    print "=> Average classifier score: {}".format(avg_clf_score)


def get_classifier():
    return sklearn.naive_bayes.GaussianNB()

class SeriesClassifier(object):
    def __init__(self, clf, vectorizer, variance_threshold, pca, inverse_document_frequency):
        self.clf = clf
        self.vectorizer = vectorizer
        self.variance_threshold = variance_threshold
        self.pca = pca
        self.inverse_document_frequency = inverse_document_frequency

    def extract_features(self, lines):
        w_f = extract_features(lines)
        for word in w_f:
            tf = w_f[word]
            idf = self.inverse_document_frequency[word]
            w_f[word] = math.log(1 + tf) * math.log(idf) if idf != 0 else 0

        features = self.vectorizer.transform([w_f])
        #features = self.variance_threshold.transform(features)
        features = features.toarray()
        #features = self.pca.transform(features)
        return features[0]

    def predict(self, features):
        return self.clf.predict(features)[0]

if __name__ == '__main__':
    shows = dict(comedy=['modern_family', '30_rock', 'big_bang_theory', 'parks_and_recreation', 'entourage'],
                 political=['house_of_cards', 'the_west_wing', 'borgen', 'the_newsroom'],
                 horror=['american_horror_story', 'penny_dreadful', 'the_walking_dead'])

    series_list = sum(shows.values(), [])

    words_frequencies = []

    series_labels = []
    subtitle_labels = []
    genre_labels = []

    for genre, genre_series in shows.iteritems():
        for series in genre_series:
            print '* {}'.format(series.upper())
            subtitles = [os.path.join(series, sub) for sub in os.listdir(series) if not sub.startswith('.')]

            for sub_path in subtitles:
                print '  . Analyzed subtitle "{}"'.format(sub_path)
                subtitle_lines = extract_lines(sub_path)
                # Some encoding errors can cause no lines to be detected
                if not subtitle_lines:
                    continue
                words_frequencies.append(extract_features(subtitle_lines))
                series_labels.append(series)
                genre_labels.append(genre)
                subtitle_labels.append(sub_path)

    series_labels, genre_labels, subtitle_labels = map(np.array, [series_labels, genre_labels, subtitle_labels])

    words_set = set(word for w_f in words_frequencies for word in w_f)

    # Calculating the inverse document frequency for each word
    inverse_document_frequency = Counter()
    total_frequency = float(sum(sum(w_f.values()) for w_f in words_frequencies))
    for word in words_set:
        total_word_frequency = len(set(series_labels[i] for i, w_f in enumerate(words_frequencies) if word in w_f))
        inverse_document_frequency[word] = math.log(len(series_list) / total_word_frequency)
        #inverse_document_frequency[word] = math.log(total_frequency / total_word_frequency)


    # Replacing word frequencies by tf-idf
    for w_f in words_frequencies:
        for word in w_f:
            tf = w_f[word]
            idf = inverse_document_frequency[word]
            w_f[word] = math.log(1 + tf) * math.log(idf) if idf != 0 else 0

    # Vectorizing the word count among all series
    vectorizer = DictVectorizer()
    feature_vectors = vectorizer.fit_transform(words_frequencies)

    # Dropping features with low variance
    MIN_VARIANCE = 0.04
    variance_threshold = VarianceThreshold(threshold=MIN_VARIANCE)
    #feature_vectors = variance_threshold.fit_transform(feature_vectors)

    # Turning to a dense matrix
    feature_vectors = feature_vectors.toarray()

    # PCA
    pca = PCA(n_components=20)
    #feature_vectors = pca.fit_transform(feature_vectors)

    # Cross-validation
    classification_validation(feature_vectors, series_labels)

    # Training an SVM classifier and dumping it to a file
    clf = get_classifier().fit(feature_vectors, series_labels)
    series_clf = SeriesClassifier(clf, vectorizer, variance_threshold, pca, inverse_document_frequency)
    with open('clf.pickle', 'w') as clf_file:
        pickle.dump(series_clf, clf_file)

    # Clustering
    CLUSTERING = False
    if CLUSTERING:
        temp_pca = PCA(n_components=4)
        X = temp_pca.fit_transform(feature_vectors)
        bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=1000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        # for i, label in enumerate(ms.labels_):
        #     print label, subtitle_labels[i]

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)

        ###############################################################################
        # Plot result
        import matplotlib.pyplot as plt
        from itertools import cycle

        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()


from collections import Counter
import os
import re
import string

import pysrt
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn import cross_validation
from sklearn import svm

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
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=0)

    clf = svm.SVC().fit(X_train, y_train)

    print "* Some predictions:"
    print "(actual_series : prediction)"
    for i, features in enumerate(X_test[:10]):
        series = y_test[i]
        prediction = clf.predict(features)[0]
        print '"{}" predicted as "{}"'.format(series, prediction)
    print "...etc."

    print "* Classifier score: {}".format(clf.score(X_test, y_test))

if __name__ == '__main__':
    shows = dict(comedy=['modern_family', '30_rock', 'big_bang_theory', 'parks_and_recreation', 'entourage'],
                 political=['house_of_cards', 'the_west_wing', 'borgen', 'the_newsroom'],
                 horror=['american_horror_story', 'penny_dreadful', 'the_walking_dead'])

    series_list = sum(shows.values(), [])

    words_frequencies = []

    series_labels = []
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


    # Vectorizing the word count among all series
    vectorizer = DictVectorizer()
    feature_vectors = vectorizer.fit_transform(words_frequencies)

    # Dropping features with low variance
    MIN_VARIANCE = 0.03
    variance_threshold = VarianceThreshold(threshold=MIN_VARIANCE)
    feature_vectors = variance_threshold.fit_transform(feature_vectors)

    # Raw features per series
    # raw_features = {series: [feature for i, feature in enumerate(feature_vectors.toarray()) if series_labels[i] == series] for series in series_list}

    # Cross-validation
    classification_validation(feature_vectors, series_labels)
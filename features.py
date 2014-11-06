from collections import Counter, defaultdict
import os
import re
import random
import string
import sys

import pysrt
import nltk

URL_REGEX = '(?:www\.)|(?:https?://)\S+\.\S+'
STOP_WORDS = set(nltk.corpus.stopwords.words('english')) | {"n't", "..."} | set(string.punctuation)
STEMMER = nltk.stem.porter.PorterStemmer()

def extract_features(subtitle_text):
    word_counter = Counter()
    for line in subtitle_lines:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            token = token.lower()
            # Skipping stop words and other unusable tokens
            if token in STOP_WORDS or token.startswith("'"):
                continue
            # Stemming
            token = STEMMER.stem(token)
            word_counter[token] += 1
    return word_counter

def extract_lines(subtitle_path):
    subtitle_object = pysrt.open(subtitle_path)
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

if __name__ == '__main__':
    series_list = ['house_of_cards', 'entourage']
    subtitles = {series: [os.path.join(series, sub) for sub in os.listdir(series)] for series in series_list}
    features_db = defaultdict(list)
    for series in series_list:
        random_sub_path = random.choice(subtitles[series])
        print '{} - {}'.format(series.upper(), random_sub_path)

        subtitle_lines = extract_lines(random_sub_path)
        #print subtitle_lines
        features = extract_features(subtitle_lines)
        print features.most_common(30)
        features_db[series].append(features)
        print
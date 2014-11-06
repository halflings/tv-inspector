from collections import Counter
import re
import string

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
    EXAMPLE_SRT = 'entourage.srt'
    subtitle_lines = extract_lines(EXAMPLE_SRT)
    #print subtitle_lines
    features = extract_features(subtitle_lines)
    print features.most_common(30)
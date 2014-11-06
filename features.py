from collections import Counter, defaultdict
import os
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

if __name__ == '__main__':
    series_list = ['house_of_cards', 'entourage']
    subtitles = {series: [os.path.join(series, sub) for sub in os.listdir(series) if not sub.startswith('.')]
                 for series in series_list}
    features_db = defaultdict(Counter)
    for series in series_list:
        print '* {}'.format(series.upper())

        for sub_path in subtitles[series]:
            print '  . Analyzed subtitle "{}"'.format(sub_path)
            subtitle_lines = extract_lines(sub_path)
            features_db[series] += extract_features(subtitle_lines)

        print
        print   '! FEATURES:'
        print
        print '\n'.join(map(str, features_db[series].most_common(50)))
        print
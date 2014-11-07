import pickle

from flask import Flask, render_template, request, jsonify
import requests

import config

# Necessary to open a pickled classifier
from features import SeriesClassifier

app = Flask(__name__)
with open('clf.pickle') as clf_file:
    series_classifier = pickle.load(clf_file)


TMDB_URL = 'http://api.themoviedb.org/3'
def get_similar_series(series):
    search_query = series.replace('_', ' ')
    res = requests.get('{}/search/tv'.format(TMDB_URL),
                       data={'api_key': config.moviedb_api_key, 'query': search_query}).json()['results'][0]
    series_id = res['id']
    similar_res =  requests.get('{}/tv/{}/similar'.format(TMDB_URL, series_id), data={'api_key': config.moviedb_api_key}).json()['results']
    return similar_res

print "* Fetching similar series..."
similar_series = {s: get_similar_series(s) for s in series_classifier.clf.classes_}

@app.route('/')
def index():
    series = series_classifier.clf.classes_
    series=  map(lambda s : s.replace('_', ' ').title(), series)
    return render_template('index.html', series=', '.join(series))

@app.route('/predict', methods=['POST'])
def predict_dialog():
    dialog = request.form['dialog']
    lines = dialog.split('\n')

    features = series_classifier.extract_features(lines)

    prediction = series_classifier.predict(features)
    return jsonify(ok=True, prediction=prediction, similar=similar_series[prediction])

if __name__ == '__main__':
    app.run(debug=True)
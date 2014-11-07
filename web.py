import pickle

from flask import Flask, render_template, request, jsonify

# Necessary to open a pickled classifier
from features import SeriesClassifier

app = Flask(__name__)
with open('clf.pickle') as clf_file:
    series_classifier = pickle.load(clf_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_dialog():
    dialog = request.form['dialog']
    lines = dialog.split('\n')

    features = series_classifier.extract_features(lines)

    prediction = series_classifier.predict(features)
    return jsonify(ok=True, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
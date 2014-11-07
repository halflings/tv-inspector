# TV Inspector

An experiment to classify TV shows using features like words used in their close-captions, etc.

# Presentation

This project was presented at Ericsson's `#ehack` hackaton.
I encourage you to [look at my slides](https://docs.google.com/presentation/d/1EVhZPhs636qVQquoG7laTnCLn3QM5F3Y4MiO4bBJGGE/edit?usp=sharing) where I go over the motivation, methodology and results of this project.

# Web app

To launch the web app, run `python web.py` (or with gunicorn: `gunicorn -b 0.0.0.0:5000 web:app`)

It looks something like this (gotta love Frank Underwood!):

![Image](../master/static/images/screenshot.png?raw=true)


# Setup

* Install dependencies: `pip install -r requirements.txt`
* Run the main file: `python features.py`



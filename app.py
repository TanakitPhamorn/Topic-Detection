import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
import cv2
import pandas as pd
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['UPLOAD_PATH'] = './static/uploads'

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def validate_text(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['chat_in']
    processed_text = make_prediction(text)
    return render_template('result.html', result = processed_text)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


def process(file):
    df = pd.read_csv(file)
    data = df[['title', 'subreddit']]
    #labels = data['subreddit']
    max_len = 0

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    tokenized = data['title'].apply((lambda x: tokenizer.encode(x,max_length=50,add_special_tokens=True, truncation=True)))
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(np.array(padded))
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    filename = 'topic_model.sav'
    infile = open(filename,'rb')
    lr_clf = pickle.load(infile)
    infile.close()

    result = lr_clf.predict(features)[0]
    return result

def make_prediction(text):
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    query = tokenizer.encode(text,max_length=50,add_special_tokens=True, truncation=True)
    query = np.array(query).reshape(1, -1)
    input_ids = torch.tensor(query)
    with torch.no_grad():
        last_hidden_states = bert_model(input_ids)
    features = last_hidden_states[0][:,0,:].numpy()
    filename = './static/topic_model.sav'
    clf_model = pickle.load(open(filename, 'rb'))
    result = clf_model.predict(features)[0]
    return result
    
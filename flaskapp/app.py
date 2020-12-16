#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
from module.nlp_cleaner import extract_hashtags
from module.nlp_cleaner import data_clean
from module.nlp_cleaner import plotly_wordcloud
from module.visualization import create_plot
from module.bert_ import load_pretrained
from module.bert_ import predict_sentiment
from module.bert_ import trans_to_sentiment
from flask import Flask, render_template, request
from flask import render_template_string, jsonify
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objs as go
import logging
from logging import Formatter, FileHandler
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
from datetime import datetime
sns.set(style="whitegrid")
#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')


#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('pages/placeholder.home.html')

@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')

@app.route('/eda')
def eda():
    date = [str(i)[:10] for i in ts.index.to_list()]
    if request.method == 'POST':
        return render_template('pages/placeholder.eda.html',
                                listStatus = date,
                                data = (min(date),max(date)))
    else:   
        select_start = request.form.get('date-select-start')
        select_end = request.form.get('date-select-end')
        return render_template('pages/placeholder.eda.html',
                                listStatus = date,
                                data = (select_start,select_end))
    # return render_template('pages/placeholder.eda.html')

@app.route('/model',methods=['GET', 'POST'])
def model():
    return render_template('pages/placeholder.model.html')
    

@app.route('/sentiment',methods=['GET', 'POST'])
def sentiment():
    sent = request.form['sent']
    prediction,prob = predict_sentiment(model,tokenizer,sent)
    label,netural,positive,negative = trans_to_sentiment(prediction,prob)
    return render_template('pages/placeholder.model.html', label = label,netural= prob[0],positive=prob[1],negative=prob[2],sentence = sent)


@app.route('/bar', methods=['GET', 'POST'])
def change_features():
    if not request.args.get("feature"):
        feature = 'Bar'
        graphJSON= create_plot(ts,feature)
    else:
        feature = request.args.get("feature")
        if feature =='Bar':
            graphJSON= create_plot(ts,feature)
        else:
            graphJSON= create_plot(ts_temp,feature)
    return graphJSON

@app.route('/topic_modelling')
def topic_modelling():
    return render_template('pages/topic_modelling.html')

@app.route('/modelpeformance')
def modelpeformance():
    return render_template('pages/modelpeformance.html')

@app.route("/api/data")
def data():
    x = [str(i)[:10] for i in ts.index.to_list()]
    return jsonify([
    {"x": x, "y": ts['Extremely Negative'].to_list(), "type": "bar","name":'Extremely Negative',"color":'#636EFA'},
    {"x": x, "y": ts.Negative.to_list(), "type": "bar","name":'Negative'},
    {"x": x,"y": ts['Neutral'].to_list(), "type": "bar","name":"Neutral"},
    {"x": x, "y": ts.Positive.to_list(), "type": "bar","name":"Positive"},
    {"x": x, "y": ts['Extremely Positive'].to_list(), "type": "bar","name":'Extremely Positive'}])


@app.route('/api/wordcloud',methods=['GET', 'POST'])
def wordcloud():
    if not request.args.get("date"):
        date = -1
    else:
        date = request.args.get("date")
    return plotly_wordcloud(df,date)




#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#


# Default port:
if __name__ == '__main__':
    path = "data/Corona_NLP_train.csv"  
    df = data_clean(path)
    ts = df.groupby(['TweetAt','Sentiment']).count()['UserName'].unstack()
    ts_temp = ts.copy()
    ts["sum"] = ts.sum(axis=1)
    ts = ts.loc[:,"Extremely Negative":"Positive"].div(ts["sum"], axis=0)
    model,tokenizer = load_pretrained()
    
    app.run()


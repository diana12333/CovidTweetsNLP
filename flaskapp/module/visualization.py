import pandas as pd
from flask import render_template_string, jsonify

def create_plot(ts,feature="Bar"):
    if feature == 'Bar':
        layout = {
                    "autosize": False,
                    "width": 1000,
                    "height": 500,
                    "margin": {
                        "l": 50,
                        "r": 50,
                        "b": 100,
                        "t": 100,
                        "pad": 4
                    },
                    "colorway" : ['#f3cec9', '#e7a4b6', '#cd7eaf', '#a262a9', '#6f4d96', '#3d3b72', '#182844'],
                    "barmode": 'stack',
                    "template":"seaborn"
                    }
        x = [str(i)[:10] for i in ts.index.to_list()]
        data = [
            {"x": x, "y": ts['Extremely Negative'].to_list(), "type": "bar","name":'Extremely Negative',"color":'#636EFA'},
            {"x": x, "y": ts.Negative.to_list(), "type": "bar","name":'Negative'},
            {"x": x, "y": ts['Neutral'].to_list(), "type": "bar","name":"Neutral"},
            {"x": x, "y": ts.Positive.to_list(), "type": "bar","name":"Positive"},
            {"x": x, "y": ts['Extremely Positive'].to_list(), "type": "bar","name":'Extremely Positive'}
        ]
        graphJSON = (jsonify([{"data":data, "layout":layout}]))
    else:
        x = [str(i)[:10] for i in ts.index.to_list()]
        layout = {
            "margin": {
                "l": 50,
                "r": 50,
                "b": 100,
                "t": 100,
                "pad": 4
            },
            "colorway" : ['#f3cec9', '#e7a4b6', '#cd7eaf', '#a262a9', '#6f4d96', '#3d3b72', '#182844'],
            "template":"seaborn"
            }
        data = [
            {"x": x, "y": ts['Extremely Negative'].to_list(), "type": "scatter","name":'Extremely Negative',"color":'#636EFA'},
            {"x": x, "y": ts.Negative.to_list(), "type": "scatter","name":'Negative'},
            {"x": x, "y": ts['Neutral'].to_list(), "type": "scatter","name":"Neutral"},
            {"x": x, "y": ts.Positive.to_list(), "type": "scatter","name":"Positive"},
            {"x": x, "y": ts['Extremely Positive'].to_list(), "type": "scatter","name":'Extremely Positive'}
        ]
        graphJSON = (jsonify([{"data":data, "layout":layout}]))

    return graphJSON
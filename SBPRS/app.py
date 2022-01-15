import model
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route("/")
def home():
    model.load_train_data()
    model.load_models()
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # getting input from HTML form
        uname = request.form.get("uname")
        exist_flag, msg = model.is_user_exist(uname)
        if (exist_flag == 0):
            return '<h2>%s</h2>' % msg
        else:
            reco_prod = model.re_suggested_prod(uname)
            final_result = model.find_positive_sentiment()

            return render_template('result.html',tables=[final_result.to_html(classes='id'), reco_prod.to_html(classes='id')], titles = ['NAN'])
    else:
        return '<h2>Page refresh is not allowed</h2>'


if __name__ == '__main__':
    app.run()
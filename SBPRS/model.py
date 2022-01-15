# utility packages
import gzip
import itertools
import os
import os.path
import pickle
import time
from itertools import groupby, product

import numpy as np
import pandas as pd
from sklearn import metrics
# model metrics related packages
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

# Author: Aniruddha Bose
 
df_train = pd.DataFrame()
sentiment_model = ""
hash_vect = ""
userbased_model = pd.DataFrame()
reco_product = pd.DataFrame()

def load_models():
    """
        Load all the models - Sentiment, Hashing Vectorizer & User-based 
    """
    # Load Sentiment Model
    filename = "HV_LR"
    foldername = "models"
    filepath = foldername + "/" + filename + ".pkl"

    global sentiment_model
    sentiment_model = pickle.load(open(filepath, 'rb'))

    # Load the Hashing vectorizer 
    filename = "hash_vect"
    foldername = "vectorizers"
    filepath = foldername + "/" + filename + ".pkl"

    global hash_vect
    hash_vect = pickle.load(open(filepath, 'rb'))

    # load the User-based model from the disk
    filename = "UB_RE"
    foldername = "models"
    filepath = foldername + "/" + filename + ".pkl"

    global userbased_model
    userbased_model = pickle.load(open(filepath, 'rb'))


def load_train_data():
    """
        Load the train data
    """
    # Read the train data
    filename = "review_train.csv"
    foldername = "datasets"
    filepath = foldername + "/" + filename

    global df_train
    df_train = pd.read_csv(filepath)
    print(df_train.shape)


def is_user_exist(uname):
    """
        Checking if a user exists or not
    """
    exist_flag = 0
    df_new = df_train[df_train["user_name"] == uname].copy()
    if len(df_new) == 0:
        msg = "User " + uname + " does not exist, please provide correct user-name"
        exist_flag = 0
        return exist_flag, msg
    else:
        exist_flag = 1
        msg = "User " + uname + " exists"
        return exist_flag, msg

def re_suggested_prod(uname):
    """
        Get Recommended System suggested products
    """
    global reco_product
    reco_product = userbased_model.loc[uname].sort_values(ascending=False)[0:20]
    reco_product = reco_product.reset_index()
    reco_product.columns = ['id','score']

    uniq_prod = df_train.drop_duplicates(['id','brand'])[['id','brand']]
    reco_product = pd.merge(reco_product,uniq_prod,left_on='id',right_on='id', how = 'left')
    return reco_product

def find_positive_sentiment():
    """
        Find positive sentiment percentage of all products suggested by the RE
    """
    # Take subset of df_train on the basis of the shortlisted 20 products suggested by the recommendation model
    df_train20 = df_train[df_train["id"].isin(reco_product["id"])]
    print(df_train20.shape)
    print(df_train20.head())

    # applying hashing vectorizer
    df_train20['r_text_lemma'] = df_train20['r_text_lemma'].fillna(' ')
    hash_vect_train20 = hash_vect.transform(df_train20['r_text_lemma'])
    # create the Document Term Matrix
    X_train20 = hash_vect_train20.toarray()
    # Prediction on train data - review text of only 20 products
    y_train20_pred = sentiment_model.predict(X_train20)

    print(X_train20.shape)
    print(y_train20_pred.shape)

    # Converting to dataframe 
    df_pred20 = pd.DataFrame(y_train20_pred)
    df_pred20.columns = ["pred_sentiment_class_num"]
    print(df_pred20.head())

    # concatenating both df_train20 and df_pred20 along columns
    df_concat = pd.DataFrame(np.hstack([df_train20,df_pred20]))
    df_concat.columns = df_train20.columns.tolist() + df_pred20.columns.tolist()
    print(df_concat.shape)
    print(df_concat.head())

    # converting to numeric field "pred_sentiment_class_num"
    df_concat["pred_sentiment_class_num"] = pd.to_numeric(df_concat["pred_sentiment_class_num"])

    # limiting to relevant columns
    df_prod_sent = df_concat[["id", "pred_sentiment_class_num"]]
    print(df_prod_sent.shape)
    print(df_prod_sent.head())

    # Calculating the positive sentiment percentage of each product
    df_sent_pct = df_prod_sent.groupby(["id"]).sum()/df_prod_sent.groupby(["id"]).count()*100
    print(df_sent_pct.shape)
    df_sent_pct = df_sent_pct.reset_index()
    df_sent_pct = df_sent_pct.rename({'pred_sentiment_class_num': 'postive_sentiment_pct'}, axis=1) 
    print(df_sent_pct.head())

    # Merging dataframe with the key as product id
    result = pd.merge(reco_product, df_sent_pct, on="id")
    print(result.shape)
    print(result.head())

    # The top 5 products with the highest percentage of positive reviews
    final_result = result.sort_values(by = ["postive_sentiment_pct"], ascending=False)[0:5]
    return final_result

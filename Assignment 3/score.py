import numpy as np
import sklearn
import pickle
import joblib
from typing import Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from scipy.sparse import load_npz

bow_msgs = load_npz('sms+spam+collection/bag_of_words.npz')

with open('sms+spam+collection/bag_of_words.pkl','rb') as f:
    bag_of_words = pickle.load(f)

tfidf_transformer = TfidfTransformer().fit(bow_msgs)

def score(text, model, threshold):

    # Ensure text is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")
    
    # Ensure threshold is between 0 and 1
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    # Get the propensity score

    try:
        text = bag_of_words.transform([text])
        # print(text.shape)
        tfidf_transformer = TfidfTransformer().fit(bow_msgs)
        text = tfidf_transformer.transform(text)
        # print(text)
        propensity = float(model.predict_proba(text)[0][1])
    except Exception as e:
        raise RuntimeError(f"Error scoring text with model: {str(e)}")
    
    # Make prediction based on threshold
    prediction = propensity >= threshold
    
    return prediction, propensity

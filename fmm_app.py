import pickle as pkl
import streamlit as st
import joblib
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd

st.set_page_config(
    page_title='NLP',
    layout="wide"
)

st.sidebar.success('Выберите нужную страницу')
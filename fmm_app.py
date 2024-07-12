import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
# from scipy.spatial.distance import chebyshev, correlation
# from sklearn.metrics.pairwise import rbf_kernel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import streamlit as st
from easygoogletranslate import EasyGoogleTranslate

# rb_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
# rb_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# distmodel = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3', token='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')

@st.cache_resource
def import_modules():
    rb_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    rb_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    distmodel = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3', token='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')
    return rb_tokenizer, rb_model, distmodel

rb_tokenizer, rb_model, distmodel = import_modules()    

def upload_and_transform():
    combined = pd.read_csv('data/combined_parsing_with_embeddings.csv')
    combined[['emb_msdisbert','emb_rubert']] = combined[['emb_msdisbert','emb_rubert']].map(lambda x: torch.FloatTensor(list(map(float,x.split(',')))))
    return combined

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

st.set_page_config(
    page_title='Предсказание'
    #layout="wide"
)

#Функция переводчик
def trsl(text):
    flag = True
    translator = EasyGoogleTranslate(
        source_language='ru',
        target_language='en',
        timeout=20
    )
    while flag:
        try:
            result = translator.translate(text)
            flag = False
        except Exception as e:
            print(e)
    return result

# Функции для обработки текста
def rubert(text,df):
    rubert_q = embed_bert_cls(text, rb_model, rb_tokenizer)
    df['sim'] = df['emb_rubert'].apply(lambda x: cosine_similarity(x.reshape(1,-1),rubert_q.reshape(1,-1))[0][0])
    return df

def distbert_ms(text,df):
    text = trsl(text)
    msdist_q = distmodel.encode(text)
    df['sim'] = df['emb_msdisbert'].apply(lambda x: cosine_similarity(msdist_q.reshape(1,-1), x.reshape(1,-1))[0][0])
    return df

def display_image(url):
    try:
        st.image(url, use_column_width=True)
    except:
        st.image('data/nope-not-here.webp', use_column_width=True)

def show_films_ds(df_final,n):
    for index, row in df_final.sort_values(by=['sim'],ascending=False).head(n).iterrows():
        cols = st.columns([1, 2])
        with cols[0]:
            display_image(row['image_url'])
        with cols[1]:
            st.write(f"**{row['movie_title']}**")
            st.write(row['description'])
            st.write(f"Сходство: {row['sim']:.4f}")

df_all = upload_and_transform()

# Заголовок страницы
st.title("Система рекомендации фильмов")

model_name = st.radio(
    "Выберите тип модели",
    ["***ru_BERT***", "***Distbert + msmarco***"])

# Форма для ввода текста
with st.form(key='text_form'):
    text_input = st.text_input(label='Введите текст')
    submit_button_one = st.form_submit_button(label='Предсказать выбранной моделью')
    #submit_button_two = st.form_submit_button(label='Предсказать моделью distbert на msmarco')

n = 10

# Обработка нажатия кнопок
if submit_button_one and model_name == "***ru_BERT***":
    result = rubert(text_input,df_all[['emb_rubert']])
    show_films_ds(pd.concat([df_all.loc[:,:'description'],result],axis=1),n)
elif submit_button_one and model_name == "***Distbert + msmarco***":
    result = distbert_ms(text_input,df_all[['emb_msdisbert']])
    show_films_ds(pd.concat([df_all.loc[:,:'description'],result],axis=1),n)
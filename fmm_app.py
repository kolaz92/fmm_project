import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import chebyshev, correlation
from sklearn.metrics.pairwise import rbf_kernel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import streamlit as st
from easygoogletranslate import EasyGoogleTranslate

rb_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
rb_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

distmodel = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3', token='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')

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
            st.write(f'Сходство: {row['sim']:.4f}')

df_all = upload_and_transform()

# Заголовок страницы
st.title("Система рекомендации фильмов")

# Форма для ввода текста
with st.form(key='text_form'):
    text_input = st.text_input(label='Введите текст')
    submit_button_one = st.form_submit_button(label='Предсказать моделью ru_BERT')
    submit_button_two = st.form_submit_button(label='Предсказать моделью distbert на msmarco')

n = 10

# Обработка нажатия кнопок
if submit_button_one:
    result = rubert(text_input,df_all[['emb_rubert']])
    show_films_ds(pd.concat([df_all.loc[:,:'description'],result],axis=1),n)
elif submit_button_two:
    result = distbert_ms(text_input,df_all[['emb_msdisbert']])
    show_films_ds(pd.concat([df_all.loc[:,:'description'],result],axis=1),n)


# # Функция для инициализации состояния сессии
# def init_session_state():
#     if 'button1' not in st.session_state:
#         st.session_state.button1 = False
#     if 'button2' not in st.session_state:
#         st.session_state.button2 = False
#     if 'button3' not in st.session_state:
#         st.session_state.button3 = False
#     if 'text_value' not in st.session_state:
#         st.session_state.text_value = ''
# # Инициализация состояния сессии
# init_session_state()

# tokenizer = nltk.RegexpTokenizer(r'\w+')

# def get_token_text(text):
#     return tokenizer.tokenize(text.lower())

# @st.cache_data
# def upload_model():
#     return joblib.load('ML_data/export_dict.pkl')

# ext_dict = upload_model()

# def preprocess_data(ext_dict):
#     dfl = []
#     models = {}
#     for key, value in ext_dict.items():
#         vect_name, model_name, model = key
#         df = pd.DataFrame(value).T
#         df = df.drop(index=['macro avg','weighted avg'],columns=['support'])
#         new_index = pd.MultiIndex.from_tuples([(f'{vect_name}_{model_name}', '0'), (f'{vect_name}_{model_name}', '1'), (f'{vect_name}_{model_name}', 'accuracy')])
#         df.index = new_index
#         dfl.append(df)
#         models[f'{vect_name}_{model_name}'] = model
#     return pd.concat(dfl), models

# def process_text(input_text, models):
#     dd = {}
#     for name, model in models.items():
#         prediction = model.predict([input_text])[0]
#         dd[name] = 'positive' if prediction else 'negative'

#     return pd.Series(dd).rename('Prediction')

# # Заголовок приложения
# st.title('Проверка отзыва на положительность или отрицательность')

# # Расположение кнопок в одну строку
# col1, col2, col3 = st.columns(3)

# # Кнопка 1
# with col1:
#     if st.button('Да'):
#         st.session_state.button1 = True

# # Кнопка 2
# with col2:
#     if st.button('Нет'):
#         st.session_state.button2 = True

# # Кнопка 3
# with col3:
#     if st.button('Сарказм'):
#         st.session_state.button3 = True

# if st.session_state.button1:
#     defvalue = '''Хочу сказать спасибо врачу, Эскулаповой А.О. Мне ампутировали ухо всего за 10 минут с анестезией, без очереди и лишних справок. С уважением, Ван Гог'''
#     st.session_state.button1 = False
# elif st.session_state.button2:
#     defvalue = '''Я пошел в поликлинику и она оказалась просто ужасной! 
#     Мне поставили диагноз шизофрения! Но я не болен, это врачи не понимают, что вырезать людям глаза - это нормально! 
#     Ужасная поликлиника!'''
#     st.session_state.button2 = False
# elif st.session_state.button3:
#     defvalue = '''Поликлиника настолько отличная, что врачи считают нормой подержать клиента 2 часа у кабинета! 
#     Обязательно обращусь туда снова, когда мне захочется бездарно потратить время!'''
#     st.session_state.button3 = False
# else:
#     defvalue = st.session_state.text_value

# st.session_state.text_value = st.text_area("Введите текст:",value=defvalue)

# summary_df, models = preprocess_data(ext_dict)
# st.dataframe(summary_df)

# # Кнопка для отправки текста в функцию
# if st.button('Отправить'):
#     # Вызов функции обработки текста
#     result = process_text(st.session_state.text_value, models)
    
#     # Вывод результата
#     st.write(st.session_state.text_value)
#     st.dataframe(result)
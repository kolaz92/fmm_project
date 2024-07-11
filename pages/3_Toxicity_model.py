import pickle as pkl
import streamlit as st
import joblib
import nltk
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title='Предсказание',
    layout="wide"
)

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()
    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

# Функция для инициализации состояния сессии
def init_session_state():
    if 'button1' not in st.session_state:
        st.session_state.button1 = False
    if 'button2' not in st.session_state:
        st.session_state.button2 = False
    if 'text_value' not in st.session_state:
        st.session_state.text_value = ''
# Инициализация состояния сессии
init_session_state()

# Заголовок приложения
deftox = 'Ну и нахрена ты сюда пришел? Тебе что, заняться больше нечем, кроме как с модельками токсичности играться? Выйди на улицу, траву потрогай, неудачник.'
defnontox = 'Ты хороший человек'
st.title('Модель проверки сообщений на токсичность')

# Расположение кнопок в одну строку
col1, col2 = st.columns(2)
defvalue = st.session_state.text_value
# Кнопка 1
with col1:
    if st.button('Токсичное'):
        st.session_state.button1 = True
        defvalue = deftox
# Кнопка 2
with col2:
    if st.button('Нетоксичное'):
        st.session_state.button2 = True
        defvalue = defnontox

if st.session_state.button1:
    defvalue = deftox
    st.session_state.button1 = False
elif st.session_state.button2:
    defvalue = defnontox
    st.session_state.button2 = False
else:
    defvalue = st.session_state.text_value

st.session_state.text_value = st.text_area("Введите текст:",value=defvalue)

# Кнопка для отправки текста в функцию
if st.button('Отправить'):
    # Вызов функции обработки текста
    result = text2toxicity(st.session_state.text_value, aggregate=True)
    
    # Вывод результата
    st.write(f'Вероятность токсичности сообщения: {result}')
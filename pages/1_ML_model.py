import pickle as pkl
import streamlit as st
import joblib
import nltk
import pandas as pd

st.set_page_config(
    page_title='Предсказание',
    layout="wide"
)

# Функция для инициализации состояния сессии
def init_session_state():
    if 'button1' not in st.session_state:
        st.session_state.button1 = False
    if 'button2' not in st.session_state:
        st.session_state.button2 = False
    if 'button3' not in st.session_state:
        st.session_state.button3 = False
    if 'text_value' not in st.session_state:
        st.session_state.text_value = ''
# Инициализация состояния сессии
init_session_state()

tokenizer = nltk.RegexpTokenizer(r'\w+')

def get_token_text(text):
    return tokenizer.tokenize(text.lower())

@st.cache_data
def upload_model():
    return joblib.load('ML_data/export_dict.pkl')

ext_dict = upload_model()

def preprocess_data(ext_dict):
    dfl = []
    models = {}
    for key, value in ext_dict.items():
        vect_name, model_name, model = key
        df = pd.DataFrame(value).T
        df = df.drop(index=['macro avg','weighted avg'],columns=['support'])
        new_index = pd.MultiIndex.from_tuples([(f'{vect_name}_{model_name}', '0'), (f'{vect_name}_{model_name}', '1'), (f'{vect_name}_{model_name}', 'accuracy')])
        df.index = new_index
        dfl.append(df)
        models[f'{vect_name}_{model_name}'] = model
    return pd.concat(dfl), models

def process_text(input_text, models):
    dd = {}
    for name, model in models.items():
        prediction = model.predict([input_text])[0]
        dd[name] = 'positive' if prediction else 'negative'

    return pd.Series(dd).rename('Prediction')

# Заголовок приложения
st.title('Проверка отзыва на положительность или отрицательность')

# Расположение кнопок в одну строку
col1, col2, col3 = st.columns(3)

# Кнопка 1
with col1:
    if st.button('Да'):
        st.session_state.button1 = True

# Кнопка 2
with col2:
    if st.button('Нет'):
        st.session_state.button2 = True

# Кнопка 3
with col3:
    if st.button('Сарказм'):
        st.session_state.button3 = True

if st.session_state.button1:
    defvalue = '''Хочу сказать спасибо врачу, Эскулаповой А.О. Мне ампутировали ухо всего за 10 минут с анестезией, без очереди и лишних справок. С уважением, Ван Гог'''
    st.session_state.button1 = False
elif st.session_state.button2:
    defvalue = '''Я пошел в поликлинику и она оказалась просто ужасной! 
    Мне поставили диагноз шизофрения! Но я не болен, это врачи не понимают, что вырезать людям глаза - это нормально! 
    Ужасная поликлиника!'''
    st.session_state.button2 = False
elif st.session_state.button3:
    defvalue = '''Поликлиника настолько отличная, что врачи считают нормой подержать клиента 2 часа у кабинета! 
    Обязательно обращусь туда снова, когда мне захочется бездарно потратить время!'''
    st.session_state.button3 = False
else:
    defvalue = st.session_state.text_value

st.session_state.text_value = st.text_area("Введите текст:",value=defvalue)

summary_df, models = preprocess_data(ext_dict)
st.dataframe(summary_df)

# Кнопка для отправки текста в функцию
if st.button('Отправить'):
    # Вызов функции обработки текста
    result = process_text(st.session_state.text_value, models)
    
    # Вывод результата
    st.write(st.session_state.text_value)
    st.dataframe(result)
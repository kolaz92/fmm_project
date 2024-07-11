import streamlit as st
from utils.preprocessing import preprocess_single_string
import json
from models.model import LSTMConcatAttention
import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F

vocab_size = 3306
embedding_dim = 64
hidden_dim = 128
output_dim = 1

with open('utils/vocab2int.json', 'r') as f:
    vocab_to_int = json.load(f)
    
@st.cache_resource
def load_model():
    clf = LSTMConcatAttention()
    clf.load_state_dict(torch.load('models/lstm_attention_weights.pt')) 
    return clf

model = load_model()

user_review = st.text_input(label='Оставьте свой отзыв здесь')
classify = st.button('Работай!!!')

if user_review and classify:
    st.write(user_review)
    preprocessed_review = preprocess_single_string(
        user_review, 
        seq_len=64, 
        vocab_to_int=vocab_to_int
    )
    st.write(preprocessed_review.shape)
    st.write(preprocessed_review)

    with torch.inference_mode():
        out = model(preprocessed_review.unsqueeze(0))
        

        
        # Предположим, out возвращает (output, другие_значения)
        output, _ = out  # Извлечение нужного значения из кортежа
        probability = torch.sigmoid(output)
        
    st.write(f'Вероятность позитивного отзыва высчитана и равна = {probability.item()}')




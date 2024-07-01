import pickle
import streamlit as st
import requests
import re
import numpy as np 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(text):
  text = re.sub('[^a-zA-Z]', ' ',text)
  tokens = word_tokenize(text.lower())
  return " ".join(WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english'))

def predict_on_user_input(input):
  user_text = preprocess(input)
  tokens = user_text.split()
  encoded_sequence = [word_index.get(token, 0) for token in tokens]  
  padded_sequence = pad_sequences([encoded_sequence], maxlen=30, padding='post')

 
  prediction = model.predict(padded_sequence)
  if prediction[0][0] > 0.5:
    return "Highly offensive and Hate speech detected"
  elif prediction[0][1] > 0.5:
    return "Hate speech detected"
  else:
    return "Positive"
  


st.header('Hate speech and Offensive language detection')

word_index = pickle.load(open('word_index.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

user_input = st.text_input("Enter the tweet:", key="user_input")

if st.button('Submit'):
    st.write(predict_on_user_input(user_input))
   

   

import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



def transfrom_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Email Finder")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transfromed_text = transfrom_text(input_sms)

    # 2. Vetorize
    vector_input = tfidf.transform([transfromed_text])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
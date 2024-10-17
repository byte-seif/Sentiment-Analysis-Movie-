import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved vectorizers and model
tfidf_vect_word = joblib.load('tfidf_vect_word.pkl')
tfidf_vect_ngram = joblib.load('tfidf_vect_ngram.pkl')
tfidf_vect_char = joblib.load('tfidf_vect_char.pkl')
lgb_model_tfidf = joblib.load('lightgbm_model_tfidf.pkl')

# Function to preprocess text input
def preprocess_text(text):
    """
    Preprocesses the input text by transforming it using the saved TF-IDF vectorizers.
    Returns a combined sparse matrix of word, n-gram, and char-level TF-IDF features.
    """
    word_tfidf = tfidf_vect_word.transform([text])
    ngram_tfidf = tfidf_vect_ngram.transform([text])
    char_tfidf = tfidf_vect_char.transform([text])
    combined_features = hstack([word_tfidf, ngram_tfidf, char_tfidf])
    return combined_features

# Streamlit UI
st.markdown(
    '<h1 style="color: orange;">Data Science and AI: Mini-Project 3</h1>',
    unsafe_allow_html=True
)
st.write("**Deployed By:** Seif Hussen")
st.write("**Description of the model deployed:**")
st.write(
    "The deployed model is a **LightGBM** classifier trained on a combination of multiple TF-IDF features, including:\n"
    "- Word-level TF-IDF vectorizer\n"
    "- N-gram TF-IDF vectorizer\n"
    "- Char-level TF-IDF vectorizer"
)
st.write("**Model Accuracy:** 87.9%")

st.title('Real-Time Movie Sentiment Analysis')
user_input = st.text_area("Enter your movie review:")

# Button for triggering sentiment analysis
if st.button('Analyze'):
    if user_input.strip():
        # Preprocess user input
        features = preprocess_text(user_input)
        # Make prediction
        prediction = lgb_model_tfidf.predict(features)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("Please enter a valid review.")

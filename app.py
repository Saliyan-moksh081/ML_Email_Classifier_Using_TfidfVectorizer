import streamlit as st
import pickle
import numpy as np

# --- Load the Saved Models ---
# We use a function with a cache decorator to load the models only once
@st.cache_resource
def load_models():
    """Loads the saved vectorizer and model from disk."""
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# --- App UI ---
st.title("Email Spam Classifier")
st.write("Enter an email text below to check if it's spam or not.")
st.write("Made by Mokshith saliyan081")

# Text area for user input
user_input = st.text_area("Email Text:", height=200)

# Predict button
if st.button("Classify Email"):
    if user_input:
        # 1. Transform the user input using the loaded vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # 2. Predict using the loaded model
        prediction = model.predict(input_tfidf)[0]
        
        # 3. Display the result
        st.subheader("Result:")
        if prediction == 1:
            st.error("This looks like SPAM.")
        else:
            st.success("This looks like HAM (Not Spam).")
    else:
        st.warning("Please enter some email text to classify.")

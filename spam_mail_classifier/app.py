# -----------------------------
# SPAM MAIL CLASSIFIER APP
# -----------------------------

import streamlit as st
import joblib

# Load saved model and TF-IDF vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Spam Mail Classifier")
st.write("Enter a message to check if it is Spam or Ham.")

# Input text
message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() != "":
        vect = vectorizer.transform([message])
        pred = model.predict(vect)[0]
        label = "Spam" if pred == 1 else "Ham"
        st.success(f"The message is classified as: **{label}**")
    else:
        st.warning("Please enter a message to classify.")

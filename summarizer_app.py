import streamlit as st
from transformers import pipeline

st.title("📄 Technical Report Summarizer")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

user_input = st.text_area("Paste a long report text here:")

if st.button("Summarize"):
    summary = summarizer(user_input, max_length=120, min_length=40, do_sample=False)
    st.subheader("Summary:")
    st.write(summary[0]['summary_text'])

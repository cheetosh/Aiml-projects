import streamlit as st
import spacy

st.title("📝 Named Entity Recognition (NER)")

nlp = spacy.load("en_core_web_sm")

user_input = st.text_area("Enter some text:")

if st.button("Extract Entities"):
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    if entities:
        st.write("Extracted Entities:")
        for ent, label in entities:
            st.write(f"**{ent}** → {label}")
    else:
        st.write("No entities found.")

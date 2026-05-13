import streamlit as st
from rag import rag_chatbot

st.title("URL RAG Chatbot")

url = st.text_input("Enter Website URL")
question = st.text_input("Ask Your Question")

if st.button("Get Answer"):

    st.write("Processing...")

    answer = rag_chatbot(url, question)

    st.write("### Response:")
    st.success(answer)
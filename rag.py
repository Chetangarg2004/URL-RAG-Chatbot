import sys
import os


sys.path.append(os.path.abspath("URL RAG Chatbot"))

from src.scraper import scrape_urls
from src.chunker import chunk_text
from src.embeddings import create_vector_store
from src.retriever import retrieve_chunks
from src.llm import generate_answer

def rag_chatbot(url, question):

    text = scrape_urls([url])

    chunks = chunk_text(text)

    vector_store = create_vector_store(chunks)

    context = retrieve_chunks(question, vector_store, chunks)

    answer = generate_answer(question, context)



    return answer
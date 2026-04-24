import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
time.sleep(1)
print("Loading... please wait")

from src.scraper import scrape_urls
from src.chunker import chunk_text
from src.embeddings import create_vector_store
from src.retriever import retrieve_chunks
from src.llm import generate_answer

def main():
    urls = input("Enter URLs (comma separated): ").split(",")

    print("Scraping content...")
    text = scrape_urls(urls)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    vector_store = create_vector_store(chunks)

    print("\nChatbot Ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        context = retrieve_chunks(query, vector_store, chunks)
        answer = generate_answer(query, context)

        print("Bot:", answer)

if __name__ == "__main__":
    main()
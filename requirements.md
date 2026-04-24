# 01-requirements.md

## Project Goal and Business Value
Build a lightweight, terminal-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer user questions based on content scraped from any given URLs.  
This tool demonstrates how to quickly turn public web pages into a smart, private knowledge-base chatbot without paying for LLMs or vector databases. It is useful for internal knowledge assistants, product demos, or quick research bots.

## User Stories
1. As a user, I can provide a list of URLs at startup so the chatbot can scrape and index their content.
2. As a user, I can have a natural conversation with the chatbot directly in the terminal.
3. As a user, I want the chatbot to retrieve relevant information from the scraped pages before answering (RAG).
4. As a user, I want the entire system to run using only free LLMs and open-source tools.

## Functional Requirements
- Accept one or more URLs as input (command-line argument or simple text file).
- Scrape readable text content from the URLs (handle basic HTML).
- Split the scraped content into chunks and build a simple RAG index (embeddings + vector store).
- Support interactive terminal chat:
  - User types a question.
  - Chatbot retrieves relevant chunks and generates an answer using the free LLM.
  - Shows the answer clearly.
- Allow the user to type "exit" to quit the chat.
- Gracefully handle cases where no relevant information is found.

## Non-Functional Requirements
- Must run completely in the terminal (no web UI).
- Use only **free** LLMs and tools (no paid API keys required).
- Keep the code simple, modular, and easy to run on a normal laptop.
- Response time should be reasonable (< 15 seconds per reply).
- No persistent database required (in-memory vector store is acceptable).

## Scope
**In-scope:**
- URL scraping (text only)
- Basic RAG pipeline (chunking → embedding → retrieval → generation)
- Terminal chat interface
- Free LLM integration

**Out-of-scope:**
- JavaScript-heavy websites or login-protected pages
- Advanced web crawling
- Saving the index to disk between runs
- Image/PDF handling
- Production deployment or authentication

## Assumptions & Dependencies
- Internet connection is available for scraping and (if needed) free LLM APIs.
- User has Python 3.10+ installed.
- The chosen free LLM must be accessible via code (Ollama, Groq, etc.).

## Success Criteria / Acceptance Tests
1. Run the program with 2–3 example URLs (e.g., a company About page + one blog post).
2. Start chatting in the terminal and ask 4–5 questions about the content of those pages.
3. The chatbot must:
   - Give accurate answers based on the scraped content.
   - Not hallucinate when information is not present.
   - Respond in natural language.
4. The program must not crash and must exit cleanly with "exit".

## Sample Input → Expected Output
**Input at startup:**

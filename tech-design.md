A terminal-based Retrieval-Augmented Generation (RAG) chatbot that answers questions from any URL using only free LLMs and open-source tool 
1. Project Overview
URLRAGChatbot is a lightweight, terminal-based chatbot that scrapes content from user-provided URLs and builds an in-memory Retrieval-Augmented Generation (RAG) pipeline. The user interacts via the terminal to ask natural language questions, and the system retrieves relevant context from the scraped pages before generating accurate answers using a free LLM.

This document covers the complete technical design for Phase 2, including architecture, tech stack justification, RAG pipeline details, LLM integration, prompt strategy, and error handling.

2. High-Level Architecture
2.1 System Flow
The pipeline follows six sequential stages from URL input to final terminal response:

  +------------------+
  |   User Input     |   (comma-separated URLs at startup)
  +--------+---------+
           |
           v
  +------------------+
  |   URL Scraper    |   (requests + BeautifulSoup)
  +--------+---------+
           |  raw HTML -> cleaned text
           v
  +------------------+
  |   Text Chunker   |   (split into ~500-token overlapping chunks)
  +--------+---------+
           |  List[str]
           v
  +------------------+
  |  Embedding Model |   (sentence-transformers: all-MiniLM-L6-v2)
  +--------+---------+
           |  List[vector]
           v
  +------------------+
  |  In-Memory Store |   (FAISS flat index)
  +--------+---------+
           |
    [Chat Loop Starts]
           |
           v
  +------------------+
  |  Query Retriever |   (embed question -> top-k similarity search)
  +--------+---------+
           |  top-3 chunks
           v
  +------------------+
  | LLM Generation   |   (Groq / Ollama) -> terminal answer
  +------------------+
2.2 Module Breakdown
Module	File	Responsibility
Scraper	src/scraper.py	Fetch URL content, strip HTML tags, return clean text
Chunker	src/chunker.py	Split text into fixed-size overlapping chunks
Embedder	src/embedder.py	Load sentence-transformer model, encode chunks and queries
VectorStore	src/vector_store.py	FAISS index: add vectors, similarity search
LLM Client	src/llm_client.py	Call Groq or Ollama API with prompt, return text
Chat Loop	src/chat.py	Terminal interaction: retrieve -> prompt -> generate
Entry Point	main.py	Parse URLs, orchestrate pipeline, start chat loop


3. Tech Stack & Justification
Component	Library / Tool	Why Chosen
Language	Python 3.10+	Industry standard for ML/NLP; rich ecosystem of free libraries
Web Scraping	requests + BeautifulSoup4	Simple, reliable; no JS rendering needed per project scope
Text Chunking	Custom (re / textwrap)	Zero dependency; full control over chunk size and overlap
Embeddings	sentence-transformers (all-MiniLM-L6-v2)	Free, runs locally on CPU; 384-dim vectors; fast inference
Vector Store	FAISS (faiss-cpu)	Facebook AI Similarity Search; free, in-memory, no DB needed
Free LLM	Groq (primary) / Ollama (fallback)	Groq: free-tier API, very fast. Ollama: fully local, no key needed
Config	python-dotenv	Load GROQ_API_KEY from .env without hardcoding credentials
Terminal UI	Built-in input() loop	No dependencies; clean REPL-style experience


4. URL Scraping
4.1 Strategy
•	Use requests.get() with a browser User-Agent header to avoid bot blocks.
•	Parse HTML with BeautifulSoup, extract only <p>, <h1>-<h6>, <li> tags.
•	Strip all scripts, styles, nav, footer elements before extraction.
•	Normalize whitespace: collapse newlines, remove empty lines.
•	Timeout: 10 seconds per URL; skip silently on failure with warning.

4.2 Sample Code
# src/scraper.py
import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    texts = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    return "\n".join(t.get_text(strip=True) for t in texts if t.get_text(strip=True))


5. RAG Pipeline Details
5.1 Chunking Strategy
Chunking splits the scraped text into overlapping windows so that context is not lost at boundaries:

Parameter	Value	Rationale
Chunk Size	500 tokens (~400 words)	Fits comfortably in LLM context while being meaningful
Overlap	50 tokens (~40 words)	Prevents answer from being split across chunk boundaries
Splitting Method	Sentence-aware (split on periods)	Avoids mid-sentence cuts for more coherent retrieval
Minimum Chunk	50 tokens	Discard very short fragments that add noise

5.2 Embedding Model
•	Model: all-MiniLM-L6-v2 from sentence-transformers
•	Output: 384-dimensional dense vectors
•	Runs 100% locally on CPU — no API key or internet needed at inference time
•	Batch-encode all chunks at startup for speed; encode queries on demand

5.3 Vector Store (FAISS)
•	Index type: IndexFlatL2 — exact nearest-neighbour search over L2 distance
•	In-memory only: rebuilt fresh every run (no persistence required per scope)
•	Top-k retrieval: k=3 most relevant chunks returned per query
•	Metadata map: maintain a dict {faiss_id -> chunk_text} for retrieval

# src/vector_store.py
import faiss, numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks: list):
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k=3) -> list:
        _, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


6. Free LLM Integration
6.1 Primary: Groq API
Groq provides a free-tier API with fast inference using open-source models. No payment required for the free tier.

Setting	Value
Model	llama3-8b-8192 (or mixtral-8x7b-32768)
API Base	https://api.groq.com/openai/v1
Auth	GROQ_API_KEY from .env
Max Tokens	512 per response
Temperature	0.3 (factual, low creativity)

6.2 Fallback: Ollama (Fully Local)
If no API key is available, Ollama runs models locally (e.g., mistral:7b). No internet required after model download.
# Install: https://ollama.ai
# Pull model:   ollama pull mistral
# Auto-detected if GROQ_API_KEY is absent


7. Prompt Strategy
7.1 System Prompt
SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based ONLY on the
provided context. If the answer is not in the context, say:
"I don't have enough information about that in the provided pages."
Do not hallucinate. Be concise and factual.
"""

7.2 User Turn with RAG Context
def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Context from scraped pages:
    {context}

    Question: {question}

    Answer based only on the context above:"""

7.3 Anti-Hallucination Strategy
•	System prompt explicitly instructs the model to refuse if context is missing.
•	Top-k=3 retrieval ensures the most relevant content is always provided.
•	Temperature set to 0.3 to reduce creativity and keep answers factual.
•	If FAISS returns 0 results (empty index), skip LLM and return a canned message.


8. Error Handling & Edge Cases
Scenario	Handling Strategy
URL unreachable / timeout	Catch requests.exceptions.RequestException, warn user, skip URL
URL returns non-200 status	Check resp.status_code, skip with warning message
Empty page (no extractable text)	Check len(text) < 50, warn user, skip URL
All URLs fail	Exit gracefully with helpful message: no knowledge base built
No relevant chunks found (k=0)	Return: "I couldn't find relevant info. Try rephrasing."
LLM API error (Groq rate limit)	Catch HTTPError, print error, prompt user to retry
Ollama not running (ConnectionRefused)	Catch ConnectionError, suggest: ollama serve in a new terminal
User types 'exit'	Break chat loop cleanly, print goodbye message
Keyboard interrupt (Ctrl+C)	Catch KeyboardInterrupt, exit gracefully


9. Project Folder Structure
URLRAGChatbot/
├── 01-requirements.md       # Phase 1 (given)
├── 02-tech-design.md        # This document (as Markdown)
├── README.md                # Setup + run instructions + screenshots
├── main.py                  # Entry point: parse URLs, run pipeline
├── requirements.txt         # All pip dependencies
├── .env.example             # GROQ_API_KEY=your_key_here
└── src/
    ├── __init__.py
    ├── scraper.py           # URL -> clean text
    ├── chunker.py           # text -> List[str] chunks
    ├── embedder.py          # chunks/query -> np.ndarray vectors
    ├── vector_store.py      # FAISS index wrapper
    ├── llm_client.py        # Groq/Ollama API caller
    └── chat.py              # Terminal chat loop


10. Setup & Run Instructions
10.1 Prerequisites
•	Python 3.10+ installed
•	pip package manager
•	(Optional) Groq free account: https://console.groq.com — get API key
•	(Optional) Ollama installed for fully local mode

10.2 Install
git clone https://github.com/your-username/URLRAGChatbot.git
cd URLRAGChatbot
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY

10.3 Run
python main.py

# You will be prompted:
Enter URLs (comma-separated): https://example.com/about, https://blog.example.com/post1

# Scraping and indexing happens automatically, then:
You: What does this company do?
Bot: Based on the About page, the company provides...

You: exit
Goodbye!


11. requirements.txt
requests>=2.31.0
beautifulsoup4>=4.12.0
sentence-transformers>=2.7.0
faiss-cpu>=1.7.4
groq>=0.9.0
python-dotenv>=1.0.0
numpy>=1.26.0


12. Success Criteria & Acceptance Tests
The following tests must pass before submission:

#	Test	Expected Result
1	Run with 2-3 URLs	Scraping completes, chat loop starts, no crash
2	Ask 4-5 factual questions	Accurate answers grounded in scraped content
3	Ask about unknown topic	System says it has no info rather than hallucinating
4	Type 'exit'	Program exits cleanly with goodbye message
5	Provide a broken URL	Warning printed, other URLs still indexed, chat continues
6	Ctrl+C during chat	Graceful exit without traceback



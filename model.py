from sentence_transformers import SentenceTransformer

model = None

def get_model():
    global model
    if model is None:
        print("Loading model ONLY ONCE...")
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return model
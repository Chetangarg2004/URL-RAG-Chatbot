from src.model import get_model
import faiss
import numpy as np

def create_vector_store(chunks):
    model = get_model()

    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index



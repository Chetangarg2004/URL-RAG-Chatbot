from src.model import get_model
import numpy as np

def retrieve_chunks(query, index, chunks, k=1):
    model = get_model()

    query_vec = model.encode(query)
    distances, indices = index.search(np.array([query_vec]), k)

    results = [chunks[i] for i in indices[0]]
    return "\n".join(results)
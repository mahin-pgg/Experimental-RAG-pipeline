import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def embed_query(question):
    return model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def search(question, top_k=3):
    query_embedding = embed_query(question)

    similarities = embeddings @ query_embedding

    ranked = np.argsort(similarities)[::-1]

    results = []

    for index in ranked[:top_k]:
        results.append({
            "score": float(similarities[index]),
            "record": records[index]
        })

    return results

BASE_DIR = Path(__file__).resolve().parent

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = np.load(
    BASE_DIR / "index" / "embeddings.npy"
)

with open(
    BASE_DIR / "index" / "records.json",
    encoding="utf-8"
) as f:
    records = json.load(f)


# TEST

question = "How do I prevent overfitting?"

results = search(question)

for result in results:
    print("\nScore:", round(result["score"], 4))
    print("Source:", result["record"]["source"])
    print(result["record"]["text"])
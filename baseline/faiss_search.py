from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Locate the saved index files relative to this script.
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"

# Load the document vectors and their matching metadata.
embeddings = np.load(INDEX_DIR / "embeddings.npy").astype(
    np.float32
)

with open(INDEX_DIR / "records.json", encoding="utf-8") as file:
    records = json.load(file)

# Each vector must correspond to exactly one record.
if len(embeddings) != len(records):
    raise ValueError("Embedding and record counts do not match")

if len(embeddings) == 0:
    raise ValueError("The index contains no document embeddings")

# Load the same model used to generate the document vectors.
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Normalize vectors so inner product equals cosine similarity.
faiss.normalize_L2(embeddings)

# Create an exact inner-product index.
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add all document embeddings to the index.
index.add(embeddings)

print("Vectors indexed:", index.ntotal)
print("Vector dimensions:", dimension)


def search(question, top_k=3):
    """Return the top matching chunks for a user question."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Convert the question into a single embedding.
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # FAISS expects a two-dimensional array:
    # (number of queries, embedding dimension).
    scores, positions = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    # Each returned position points to its matching metadata record.
    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue

        results.append({
            "score": float(score),
            "record": records[int(position)],
        })

    return results


if __name__ == "__main__":
    question = "How can I reduce overfitting?"

    for rank, result in enumerate(search(question), start=1):
        print(f"\nRank: {rank}")
        print(f"Cosine similarity: {result['score']:.4f}")
        print(f"Source: {result['record']['source']}")
        print(result["record"]["text"])

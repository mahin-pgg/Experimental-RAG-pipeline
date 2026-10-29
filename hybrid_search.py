import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Our miniature knowledge base
# --------------------------------------------------

documents = [
    {
        "id": "doc_1",
        "text": "Error E-1042 occurs when the database connection times out."
    },
    {
        "id": "doc_2",
        "text": "To fix database connection failures, check the server and network."
    },
    {
        "id": "doc_3",
        "text": "Reset your password from the account settings page."
    },
    {
        "id": "doc_4",
        "text": "The FindBack application uses semantic embeddings to match lost items."
    },
    {
        "id": "doc_5",
        "text": "A database timeout can happen when the database server is overloaded."
    },
]

texts = [doc["text"] for doc in documents]


# --------------------------------------------------
# 2. Simple tokenizer for lexical retrieval
# --------------------------------------------------

def tokenize(text):
    """
    Lowercase the text and extract word-like tokens.
    Keep internal hyphens so E-1042 remains one token.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


# --------------------------------------------------
# 3. Build the BM25 index
# --------------------------------------------------

tokenized_documents = [tokenize(text) for text in texts]

bm25 = BM25Okapi(tokenized_documents)


# --------------------------------------------------
# 4. Build the dense retrieval index
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Normalize embeddings so dot products equal cosine similarity.
document_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)


# --------------------------------------------------
# 5. Retrieve using BM25
# --------------------------------------------------

def lexical_search(query):
    query_tokens = tokenize(query)

    # One BM25 score per document.
    scores = bm25.get_scores(query_tokens)

    # Sort document positions by descending score.
    ranked_ids = np.argsort(scores)[::-1]

    return [
        (int(i), float(scores[i]))
        for i in ranked_ids
        if scores[i] > 0
    ]


# --------------------------------------------------
# 6. Retrieve using dense embeddings
# --------------------------------------------------

def dense_search(query):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Shape: (number of documents,)
    scores = document_embeddings @ query_embedding

    ranked_ids = np.argsort(scores)[::-1]

    return [
        (int(i), float(scores[i]))
        for i in ranked_ids
    ]


# --------------------------------------------------
# 7. Combine rankings with Reciprocal Rank Fusion
# --------------------------------------------------

def hybrid_search(query, top_k=3, rrf_k=60):
    lexical_results = lexical_search(query)
    dense_results = dense_search(query)

    fused_scores = {}

    # Add contributions from the lexical ranking.
    for rank, (doc_id, _) in enumerate(lexical_results, start=1):
        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    # Add contributions from the dense ranking.
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        fused_scores[doc_id] = (
            fused_scores.get(doc_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    # Rank by the combined score.
    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    return [
        (doc_id, fused_scores[doc_id])
        for doc_id in ranked_ids[:top_k]
    ]


# --------------------------------------------------
# 8. Compare all three methods
# --------------------------------------------------

query = "what causes database connection timeouts?"

print("\n--- BM25 / Lexical Search ---")
for doc_id, score in lexical_search(query)[:3]:
    print(f"{score:.4f} | {documents[doc_id]['text']}")

print("\n--- Dense / Semantic Search ---")
for doc_id, score in dense_search(query)[:3]:
    print(f"{score:.4f} | {documents[doc_id]['text']}")

print("\n--- Hybrid Search (RRF) ---")
for doc_id, score in hybrid_search(query, top_k=3):
    print(f"{score:.4f} | {documents[doc_id]['text']}")

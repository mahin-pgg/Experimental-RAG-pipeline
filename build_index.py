from ingestion import ingest_text_folder
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

records = ingest_text_folder(BASE_DIR / "documents")

texts = [record["text"] for record in records]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(
    INDEX_DIR / "embeddings.npy",
    embeddings
)

with open(
    INDEX_DIR / "records.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Indexed {len(records)} chunks.")

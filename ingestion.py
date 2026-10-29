from pathlib import Path
import re

# Resolve the document directory relative to this script.
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "documents"


def load_text(file_path):
    """Read a UTF-8 text document."""
    return file_path.read_text(encoding="utf-8")


def clean_text(text):
    """Normalize common whitespace problems."""

    # Replace Windows and old-style line endings with '\n'.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs and repeated spaces with one space.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the start and end of each line.
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Keep paragraph breaks but collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Load one document and inspect it.
file_path = DOCUMENT_DIR / "ml_notes.txt"

raw_text = load_text(file_path)
cleaned_text = clean_text(raw_text)

print(cleaned_text)

def sentence_chunks(text, chunk_size=300, overlap_sentences=1):
    """
    Group complete sentences into chunks.

    chunk_size is a target character length, not a strict limit.
    overlap_sentences controls how many trailing sentences
    are carried into the next chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    # Split on whitespace following common sentence-ending
    # punctuation. This is a simple baseline, not a full
    # linguistic sentence tokenizer.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        # Account for a space between existing sentences.
        extra_length = len(sentence)
        if current:
            extra_length += 1

        # If adding this sentence exceeds the target,
        # save the existing chunk first.
        if current and current_length + extra_length > chunk_size:
            chunks.append(" ".join(current))

            # Carry over a few trailing sentences.
            if overlap_sentences:
                current = current[-overlap_sentences:]
            else:
                current = []

            # Recalculate length after overlap.
            current_length = len(" ".join(current))

        # Add the new sentence to the current chunk.
        if current:
            current_length += 1

        current.append(sentence)
        current_length += len(sentence)

    # Do not forget the final chunk.
    if current:
        chunks.append(" ".join(current))

    return chunks

def build_chunk_records(text, source_name, chunk_size=300,
                        overlap_sentences=1):
    """
    Create structured records from sentence-based chunks.
    """

    texts = sentence_chunks(
        text,
        chunk_size=chunk_size,
        overlap_sentences=overlap_sentences,
    )

    records = []

    for index, chunk_text in enumerate(texts):
        record = {
            "chunk_id": f"{source_name}_chunk_{index}",
            "document_id": source_name,
            "source": source_name,
            "chunk_index": index,
            "char_count": len(chunk_text),
            "text": chunk_text,
        }

        records.append(record)

    return records


records = build_chunk_records(
    cleaned_text,
    source_name="ml_notes",
    chunk_size=300,
    overlap_sentences=1,
)

for record in records:
    print(record)

def ingest_text_folder(folder_path):
    """Load supported text files and create chunk records."""

    all_records = []

    # Find .txt and .md files in the folder.
    files = sorted(
        list(folder_path.glob("*.txt"))
        + list(folder_path.glob("*.md"))
    )

    for file_path in files:
        # 1. Extract the file's text.
        raw = load_text(file_path)

        # 2. Clean the extracted text.
        cleaned = clean_text(raw)

        # 3. Split the text and attach source metadata.
        records = build_chunk_records(
            cleaned,
            source_name=file_path.name,
            chunk_size=300,
            overlap_sentences=1,
        )

        all_records.extend(records)

    return all_records


if __name__ == "__main__":
    records = ingest_text_folder(DOCUMENT_DIR)

    print(f"Created {len(records)} chunks.")

    for record in records:
        print(
            f"{record['chunk_id']} | "
            f"{record['char_count']} characters"
        )
from langchain_community.document_loaders import (TextLoader)

def load_documents(file_path: str):
    """
    Load documents from a text file.

    Args:
        file_path (str): The path to the text file.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

docs = load_documents("documents/ml_notes.txt")

print(docs[0].page_content)
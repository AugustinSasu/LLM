import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import EmbeddingFunction
import requests
import numpy as np


# Using Ollama instead of OpenAI since OpenAI key was not available :)
class OllamaEmbeddingFunction(EmbeddingFunction):
    # Ollama uses nomic-embed-text; OpenAI uses text-embedding-3-small
    def __init__(self, model="nomic-embed-text"):
        self.model = model

    def __call__(self, texts):
        return [self._normalize(self.get_embedding(text)) for text in texts]

    def get_embedding(self, text):
        response = requests.post(
            # the local server address where Ollama runs
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    # the embeddings should be normalized
    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return [x / norm for x in vec] if norm != 0 else vec


def query_ollama(model_name, user_prompt):
    # the local server address where Ollama runs
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": user_prompt, "stream": False}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]


# extracting the titles and summaries
def load_books(path="data/books.txt"):
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    # split() does not include the separator in the resulting strings
    books = content.split("## Title: ")[1:] # [1:] to skip the first element of the "books" list because it's an empty string
    summaries = []
    titles = []

    for item in books:
        lines = item.strip().split("\n")

        title = lines[0].strip()
        summary = " ".join(line.strip() for line in lines[1:] if line.strip())

        titles.append(title)
        summaries.append(summary)

    return titles, summaries


def create_vector_store(titles, summaries):
    # connects to a local ChromaDB database
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OllamaEmbeddingFunction()

    # get a list of all collection names already stored in the db
    existing_collections = [c.name for c in client.list_collections()]

    # If the collection "books" exists, delete and recreate to ensure fresh data
    if "books" in existing_collections:
        client.delete_collection("books")

    # Create the collection
    collection = client.create_collection("books", embedding_function=embedding_fn)

    # Add documents to the "books" collection
    collection.add(
        documents=summaries,
        ids=[f"book{i}" for i in range(len(titles))],
        metadatas=[{"title": title} for title in titles]
    )
    return collection


# test_search needs it's own client and embedding function to access and search
# to use this change the path at line 37 to ../data/books.txt and app.tools instead of tools
def test_search(query: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OllamaEmbeddingFunction()
    collection = client.get_collection("books", embedding_function=embedding_fn)

    results = collection.query(query_texts=[query], n_results=3)
    print("\n Query:", query)
    for title, doc in zip(results["metadatas"][0], results["documents"][0]):
        print(f"\n Title: {title['title']}\n Summary: {doc}")


if __name__ == "__main__":
    titles, summaries = load_books()
    create_vector_store(titles, summaries)
    test_search("magic and friendship")
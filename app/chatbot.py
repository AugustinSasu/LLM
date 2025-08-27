import chromadb
from app.rag import OllamaEmbeddingFunction


def recommend_book(user_input):
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OllamaEmbeddingFunction()
    collection = client.get_collection("books", embedding_function=embedding_fn)

    # collection.query search the ChromaDB vector store and returns top n_results most similar
    results = collection.query(query_texts=[user_input], n_results=3)

    if not results["metadatas"] or not results["metadatas"][0]:
        return "No match found", "Sorry, I couldn't find a book matching your interests."

    match = results["metadatas"][0][0]  # Get the metadata of the top match
    matched_title = match['title']

    # Return result
    return matched_title
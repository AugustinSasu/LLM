import chromadb
from app.rag import OllamaEmbeddingFunction


def recommend_book(user_input):
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OllamaEmbeddingFunction()
    collection = client.get_collection("books", embedding_function=embedding_fn)

    # collection.query search the ChromaDB vector store and returns top n_results most similar
    # results is a dictionary
    results = collection.query(query_texts=[user_input], n_results=1, include=["metadatas", "distances"])

    # results["metadatas"] is a list of lists: one list per query
    # results["metadatas"][0] is the list of metadatas for the first query
    # [
    #     [ {'title': 'Book 3'}, {'title': 'Book 1'}, {'title': 'Book 7'} ]
    # ]
    if not results["metadatas"] or not results["metadatas"][0]:
        return "No match found", "Sorry, I couldn't find a book matching your interests."

    # distances refers to Euclidean distance; smaller distance means more similar
    top_distance = results["distances"][0][0]
    print(top_distance)

    # if the title isn't related enough return None
    if top_distance > 0.85:
        return None
    
    # results["metadatas"][0][0] takes the first dictionary from the first query
    match = results["metadatas"][0][0]
    matched_title = match['title']

    # Return result
    return matched_title
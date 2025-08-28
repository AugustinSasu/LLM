from app.chatbot import recommend_book
from app.rag import load_books, create_vector_store, query_ollama
from app.tools import build_summary_dict, get_summary_by_title

titles, summaries = load_books()
book_list_str = "\n".join([f"- {title}" for title in titles])
create_vector_store(titles, summaries)
build_summary_dict(titles, summaries)

model_name = "gpt-oss:20b"
conversation_history = [
    {"role": "system", "content": (
        "You are a helpful and conversational librarian. "
        "Only recommend books from the list provided. "
        "Never make up titles. Provide summaries only from the provided data. "
        "Be friendly, concise, and informative."
    )}
]

print("Welcome to the Smart Librarian!")
print("Ask me for a book recommendation based on your interests.")

while True:
    user_input = input("\n You: ").strip()
    if user_input.lower() == "exit":
        break
    
    for title in titles:
        if title.lower() in user_input.lower() and "summary" in user_input.lower():
            summary = get_summary_by_title(title)
            print(f"Librarian: Here's the summary for '{title}':\n{summary}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": summary})
            break
    else:
        matched_title = recommend_book(user_input)
        if matched_title is None:
            response = "Sorry, I couldn't find a book matching your interests."
            print(f"Librarian: {response}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            continue  # loop again if no matched_title
  
    # after LLM makes a recommendation, we can provide a summary using get_summary_by_title tool
    detailed_summary = get_summary_by_title(matched_title)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({
        "role": "assistant",
        "content": (
            f"I recommend the book '{matched_title}'. "
            "Here's the original summary from our records (do not rephrase it):\n"
            f"{detailed_summary}"
        )
    })

    prompt = (
        "Available books:\n" + book_list_str + "\n\n"
        "Here is the conversation so far:\n" +
        "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history]) +
        "\nUser: "
    )

    # Get conversational response from Ollama
    ollama_response = query_ollama(model_name, prompt)
    print(f"Librarian: {ollama_response}")

    # Update conversation history
    conversation_history.append({"role": "assistant", "content": ollama_response})
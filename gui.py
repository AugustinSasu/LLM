import tkinter as tk
from tkinter import scrolledtext

from typer import prompt
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


def send_message():
    user_input = user_entry.get()
    if not user_input.strip():
        return
    
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_input}\n")
    user_entry.delete(0, tk.END)

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
            chat_window.insert(tk.END, f"Librarian: {response}\n")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            chat_window.config(state=tk.DISABLED)
            return

    detailed_summary = get_summary_by_title(matched_title.strip())

    prompt = (
        f"You are a helpful librarian. Respond to the user in one friendly sentence "
        f"recommending the book '{matched_title}' without rephrasing its summary.\n"
        f"User: {user_input}"
    )
    ollama_response = query_ollama(model_name, prompt)

    chat_window.insert(tk.END, f"Librarian: {ollama_response}\n")

    # Show original summary separately
    chat_window.insert(tk.END, f"\nSummary:\n{detailed_summary}\n")

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": f"{ollama_response}\n{detailed_summary}"})

    chat_window.config(state=tk.DISABLED)


# GUI setup
root = tk.Tk()
root.title("Smart Librarian Chatbot")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=20)
chat_window.pack(padx=10, pady=10)

user_entry = tk.Entry(root, width=70)
user_entry.pack(side=tk.LEFT, padx=(10,0), pady=(0,10), expand=True, fill=tk.X)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=(5,10), pady=(0,10))

root.mainloop()

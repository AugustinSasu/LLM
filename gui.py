# for GUI
import tkinter as tk
from tkinter import scrolledtext
from typer import prompt

# for text to speech
import pyttsx3
# for speech to text
import sounddevice as sd
import queue
import vosk
import json

# for words filter
from better_profanity import profanity
profanity.load_censor_words()

from app.chatbot import recommend_book
from app.rag import load_books, create_vector_store, query_ollama
from app.tools import build_summary_dict, get_summary_by_title


engine = pyttsx3.init()
# default rate is 200 which is quite fast
engine.setProperty("rate", 150)
# true means speech is on
tts_enabled = True

def toggle_tts():
    global tts_enabled
    tts_enabled = not tts_enabled
    tts_button.config(text="🔈 Unmute" if not tts_enabled else "🔇 Mute")

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def listen_offline():
    model = vosk.Model("models/vosk-model-small-en-us-0.15")
    samplerate = 16000
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result.get("text", "")

def handle_mic_input():
    chat_window.config(state=tk.NORMAL) # NORMAL allows text to be inserted by user
    chat_window.insert(tk.END, "🎤 Listening...\n") # END means inserting after the last character
    chat_window.config(state=tk.DISABLED) # user cannot edit anymore

    chat_window.see(tk.END)
    
    root.update()  # Force the GUI to update right now

    # Now do the actual listening
    text = listen_offline()

    chat_window.config(state=tk.NORMAL)
    chat_window.config(state=tk.DISABLED)
    chat_window.see(tk.END)
    user_entry.delete(0, tk.END)
    user_entry.insert(0, text)



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
    
    # check the user's input for unallowed words
    if profanity.contains_profanity(user_input):
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "Librarian: Please avoid inappropriate language.\n")
        chat_window.config(state=tk.DISABLED)
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
        f"recommending the book '{matched_title}' without rephrasing its summary and give the correct authors\n"
        f"User: {user_input}"
    )

    ollama_response = query_ollama(model_name, prompt)
    
    chat_window.insert(tk.END, f"Librarian: {ollama_response}\n")

    # Show original summary separately
    chat_window.insert(tk.END, f"\nSummary:\n{detailed_summary}\n")

    root.update()

    if tts_enabled:
        engine.say(ollama_response)
        engine.say(detailed_summary)
        # text won't be said without this
        engine.runAndWait()

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": f"{ollama_response}\n{detailed_summary}"})

    chat_window.config(state=tk.DISABLED)


# GUI setup
root = tk.Tk()
root.title("Smart Librarian Chatbot")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=20)
chat_window.pack(padx=10, pady=10)

mic_button = tk.Button(root, text="🎙️ Speak", command=handle_mic_input)
mic_button.pack(side=tk.LEFT, padx=(5,10), pady=(0,10))

tts_button = tk.Button(root, text="🔇 Mute", command=toggle_tts)
tts_button.pack(side=tk.LEFT, padx=(5, 10), pady=(0, 10))

user_entry = tk.Entry(root, width=70)
user_entry.pack(side=tk.LEFT, padx=(10,0), pady=(0,10), expand=True, fill=tk.X)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=(5,10), pady=(0,10))



root.mainloop()

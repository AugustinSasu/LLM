from app.chatbot import recommend_book
from app.rag import load_books, create_vector_store
from app.tools import build_summary_dict, get_summary_by_title

titles, summaries = load_books()
create_vector_store(titles, summaries)
build_summary_dict(titles, summaries)


print("Welcome to the Smart Librarian!")
print("Ask me for a book recommendation based on your interests.")

while True:
    user_input = input("\n You: ").strip()
    if user_input.lower() == "exit":
        break

    matched_title = recommend_book(user_input)
    if matched_title is None:
        print("Sorry, I couldn't find a book matching your interests.")
    else:
        print(f"\n Recommended Book: {matched_title}")
        # after LLM makes a recommendation, we can provide a summary using get_summary_by_title tool
        detailed_summary = get_summary_by_title(matched_title)
        print(f" Summary: {detailed_summary}")
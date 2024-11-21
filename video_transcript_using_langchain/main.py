from database.config import DB_CONNECTION
from langchain_groq import ChatGroq
from transcript.youtue_transcript import fetch_and_save_transcript
from vectore_store.vectore_store import initialize_vectorstore
from retreiver.retreiver import create_retriever, create_rag_chain
from prompts.prompts import get_qa_prompt
from state_management.state_management import initialize_workflow
from langchain.chains.combine_documents import create_stuff_documents_chain
from chat_history import initialize_chat_history, get_recent_chat_history, add_message_to_history


def main():
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)

    video_id = input("Give youtube video_id: ")
    # Fetch and process transcript
    transcript_file = fetch_and_save_transcript(video_id)

    # Initialize vectorstore
    vectorstore = initialize_vectorstore(transcript_file, "transcript", DB_CONNECTION)

    chat_history = initialize_chat_history()

    # Create retriever and RAG chain
    retriever = create_retriever(llm, vectorstore, chat_history)
    qa_prompt = get_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_rag_chain(retriever, question_answer_chain)

    # Initialize chatbot workflow
    app = initialize_workflow(rag_chain)

    print("Chatbot initialized. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        add_message_to_history(chat_history, "human", user_input)

        # Retrieve context from vector store using invoke
        context = retriever.invoke({"input": user_input, "chat_history": get_recent_chat_history(chat_history)})
        # print("Context Retrieved:", context)

        combined_input = {
            "input": user_input,
            "chat_history": get_recent_chat_history(chat_history),
            "context": context
        }
        # print("Combined Input for RAG:", combined_input)
        # Update the config dictionary to include required keys
        config = {
            "configurable": {
                "thread_id": "abc123",  # Provide a unique thread ID
                "checkpoint_ns": "your_namespace",  # Provide a namespace if needed
                "checkpoint_id": "your_checkpoint_id"  # Provide a checkpoint ID if needed
            }
        }

        result = app.invoke(combined_input, config=config)
        print(f"AI: {result['answer']}")

if __name__ == "__main__":
    main()

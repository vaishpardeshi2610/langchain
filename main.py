import asyncio
from database.db_operations import load_env_variables, get_db_params, DatabaseManager
from transcripts.transcript_operations import fetch_transcript, split_into_sentences
from embeddings_store.embedding_operations import EmbeddingUtils
from groq_services.chat_operations import ChatWorkflow, count_tokens, truncate_input
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage

async def main():
    load_env_variables()
    db_params = get_db_params()

    # Initialize components
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embedding_utils = EmbeddingUtils(embedding_model) #creating an object of the embedding model
    db_manager = DatabaseManager(db_params) #Manages PostgreSQL connections and operations and creats an object of the database manager
    chat_workflow = ChatWorkflow()

    # Fetch transcript
    video_id = input("Enter the YouTube video ID: ")
    transcript = fetch_transcript(video_id)
    if not transcript:
        return

    # Generate and store embeddings
    embedding = embedding_utils.generate_transcript_embedding(transcript)
    await db_manager.store_embedding(video_id, transcript, embedding) # This line stores the embeddings in the database. The database operation is I/O-bound, and using await allows the program to remain responsive while waiting for the database operation to complete.

    # Start chat
    print("\nStart chatting about the video! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        relevant_sections = embedding_utils.find_relevant_section(user_input, transcript)
        if relevant_sections:
            relevant_text = " ".join([sent for sent, _ in relevant_sections])

            # Check the token count and truncate if necessary
            if count_tokens(relevant_text) > 5000:  # Adjust the limit as needed
                relevant_text = truncate_input(relevant_text, max_tokens=5000)

            input_message = HumanMessage(content=relevant_text)

            # Check the token count of the input message before sending
            if count_tokens(relevant_text) > 5000:
                print("Message too large, truncating...")
                relevant_text = truncate_input(relevant_text, max_tokens=5000)
                input_message = HumanMessage(content=relevant_text)

            # Send the relevant section to the model for processing
            try:
                # Assuming you have a function to handle this
                response = await chat_workflow.process_input(input_message)
                print("Response:", response)
            except Exception as e:
                print("Error processing input:", e)
        else:
            print("No relevant sections found.")

if __name__ == "__main__":
    asyncio.run(main())

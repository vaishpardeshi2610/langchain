from langchain.chains.history_aware_retriever import create_history_aware_retriever #A function that creates a retriever capable of considering historical context when generating responses.
from langchain.chains.retrieval import create_retrieval_chain #A function that creates a chain for retrieval-augmented generation (RAG), which combines retrieval and generation processes.
from langchain_core.prompts import ChatPromptTemplate #A class used to create structured prompts for chat interactions.


def create_retriever(llm, vectorstore, chat_history):
    """Creates a retrieval-aware chain."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}) #The as_retriever method is called on the vectorstore instance to create a retriever. This retriever will perform similarity searches to find the most relevant documents based on the user's query. The search_kwargs parameter specifies that the retriever should return the top 6 most similar documents.
    
    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    return history_aware_retriever  # Return the history-aware retriever directly

def create_rag_chain(history_aware_retriever, question_answer_chain):
    """Creates a Retrieval-Augmented Generation chain."""
    return create_retrieval_chain(history_aware_retriever, question_answer_chain) # combines the history-aware retriever with the question-answering chain. This allows the system to retrieve relevant information and generate answers based on that information.

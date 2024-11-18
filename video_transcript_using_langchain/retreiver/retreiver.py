from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def create_retriever(llm, vectorstore):
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

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def create_rag_chain(history_aware_retriever, question_answer_chain):
    """Creates a Retrieval-Augmented Generation chain."""
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

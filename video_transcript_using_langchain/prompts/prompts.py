from langchain_core.prompts import ChatPromptTemplate

def get_qa_prompt():
    """Returns the QA prompt template."""
    system_prompt = (
    "You are an assistant for question-answering tasks. Use ONLY the retrieved context below "
    "to answer the question. Do not use external knowledge or guess. If the context is insufficient, "
    "respond with 'I don't know.'\n\n"
    "Retrieved context:\n{context}\n"
    "Question:"
)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    input: str
    chat_history: Annotated[list, add_messages]
    context: str
    answer: str

def call_model(state, rag_chain):
    """Handles the call to the RAG chain."""
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

def initialize_workflow(rag_chain):
    """Initializes and returns the workflow."""
    workflow = StateGraph(state_schema=State) 
    workflow.add_edge(START, "model")
    workflow.add_node("model", lambda state: call_model(state, rag_chain))

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

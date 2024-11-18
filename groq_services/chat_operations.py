import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

class ChatWorkflow:
    def __init__(self, model_name="mixtral-8x7b-32768"):
        self.memory = MemorySaver()
        self.workflow = StateGraph(state_schema=MessagesState)
        self.chat_model = ChatGroq(temperature=0, model_name=model_name)
        self._setup_workflow()

    def _setup_workflow(self):
        def call_model(state: MessagesState):
            response = self.chat_model.invoke(state["messages"])
            return {"messages": response}

        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": uuid.uuid4()}}

    def process_input(self, input_message):
        for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
            event["messages"][-1].pretty_print()

def count_tokens(text):
    """
    Counts the number of tokens in the input text.
    This is a simplified version; consider using a proper tokenizer for accuracy.
    """
    return len(text.split())

def truncate_input(input_text, max_tokens=5000):
    """
    Truncates the input text to ensure it fits within the token limit.
    Here, we assume tokens are separated by spaces.
    """
    tokens = input_text.split()  # Simplified tokenization by splitting on spaces
    return ' '.join(tokens[:max_tokens])  # Return only the first 'max_tokens' tokens
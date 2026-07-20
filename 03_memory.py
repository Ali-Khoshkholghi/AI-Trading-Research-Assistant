from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 2. Setup LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 3. Define Node
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

# 5. Add memory (the key new part!)
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 6. Config — thread_id is like a conversation ID
config = {"configurable": {"thread_id": "ali_conversation_1"}}

# 7. First message
print("--- Turn 1 ---")
result = app.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Ali and I am learning LangGraph.")]},
    config=config
)
print(result["messages"][-1].content)

# 8. Second message — same thread_id = same memory!
print("\n--- Turn 2 ---")
result = app.invoke(
    {"messages": [HumanMessage(content="What is my name and what am I learning?")]},
    config=config
)
print(result["messages"][-1].content)
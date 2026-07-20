from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

# 2. Define LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 3. Define Nodes
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def check_response(state: AgentState):
    last = state["messages"][-1].content
    if "DONE" in last:
        return {"next_step": "end"}
    return {"next_step": "llm"}

# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("check", check_response)
graph.set_entry_point("llm")
graph.add_edge("llm", "check")
graph.add_conditional_edges(
    "check",
    lambda state: state["next_step"],
    {"llm": "llm", "end": END}
)
app = graph.compile()

# 5. Run it
result = app.invoke({
    "messages": [HumanMessage(content="Write a short poem and say DONE at the end.")],
    "next_step": "llm"
})

print(result["messages"][-1].content)
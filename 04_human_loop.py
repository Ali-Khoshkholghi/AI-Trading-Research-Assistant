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
    draft: str
    approved: bool

# 2. Setup LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 3. Define Nodes
def write_draft(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"draft": response.content}

def human_review(state: AgentState):
    print("\n--- DRAFT EMAIL ---")
    print(state["draft"])
    print("-------------------")
    # Simple Python input — waits for you to type!
    decision = input("\nDo you approve this email? (yes/no): ")
    return {"approved": decision.lower() == "yes"}

def send_email(state: AgentState):
    print("\n✓ Email approved and sent!")
    return {}

def reject_email(state: AgentState):
    print("\n✗ Email rejected. Discarded.")
    return {}

# 4. Routing
def route_decision(state: AgentState):
    if state["approved"]:
        return "send"
    return "reject"

# 5. Build Graph
graph = StateGraph(AgentState)
graph.add_node("write", write_draft)
graph.add_node("review", human_review)
graph.add_node("send", send_email)
graph.add_node("reject", reject_email)

graph.set_entry_point("write")
graph.add_edge("write", "review")
graph.add_conditional_edges(
    "review",
    route_decision,
    {"send": "send", "reject": "reject"}
)
graph.add_edge("send", END)
graph.add_edge("reject", END)

# 6. Compile
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 7. Run
config = {"configurable": {"thread_id": "email_1"}}

app.invoke(
    {
        "messages": [HumanMessage(content="Write a short professional email to my team announcing we are adopting LangGraph.")],
        "draft": "",
        "approved": False
    },
    config=config
)
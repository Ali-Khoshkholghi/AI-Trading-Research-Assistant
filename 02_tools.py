from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# 1. Define a tool
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Input should be a math expression like '2 + 2' or '10 * 5'."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# 2. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 3. Setup LLM with tools bound to it
tools = [calculator]
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# 4. Define Nodes
def call_llm(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    result = calculator.invoke(tool_call["args"])
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}

# 5. Routing function
def should_use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"

# 6. Build Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool", call_tool)
graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_use_tool,
    {"use_tool": "tool", "end": END}
)
graph.add_edge("tool", "llm")
app = graph.compile()

# 7. Run it
result = app.invoke({
    "messages": [HumanMessage(content="What is 1234 * 5678?")]
})

print(result["messages"][-1].content)
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# 1. Real web search tool
search = TavilySearch(max_results=5)

# 2. Shared State
class AgentState(TypedDict):
    topic: str
    research: str
    draft: str
    score: int
    attempts: int

# 3. LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 4. Agent 1 — Researcher (now uses real web search!)
def researcher(state: AgentState):
    print(f"\n🔍 Researcher: searching the web for '{state['topic']}'...")
    results = search.invoke(state["topic"])
    print(f"   Found real data!")
    return {"research": results}

# 5. Agent 2 — Writer
def writer(state: AgentState):
    print("\n✍️  Writer: writing the blog post...")
    response = llm.invoke([
        SystemMessage(content="You are a blog post writer. Write a short, engaging blog post based ONLY on the research provided. Do not add any facts not in the research."),
        HumanMessage(content=f"Topic: {state['topic']}\n\nReal research data:\n{state['research']}")
    ])
    return {"draft": response.content, "attempts": state["attempts"] + 1}

# 6. Agent 3 — Reviewer
def reviewer(state: AgentState):
    print("\n📝 Reviewer: reviewing the draft...")
    response = llm.invoke([
        SystemMessage(content="You are a strict blog post reviewer. Score the blog post from 1-10. Reply with ONLY a number."),
        HumanMessage(content=f"Review this blog post:\n\n{state['draft']}")
    ])
    try:
        score = int(response.content.strip())
    except:
        score = 7
    print(f"   Score: {score}/10")
    return {"score": score}

# 7. Routing
def route_review(state: AgentState):
    if state["score"] >= 7 or state["attempts"] >= 3:
        return "approved"
    print("\n   Score too low! Sending back to writer...")
    return "rewrite"

# 8. Build Graph
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges(
    "reviewer",
    route_review,
    {"approved": END, "rewrite": "writer"}
)

app = graph.compile()

# 9. Run it
topic = input("What topic do you want the blog post about? ")

result = app.invoke({
    "topic": topic,
    "research": "",
    "draft": "",
    "score": 0,
    "attempts": 0
})

print("\n" + "="*50)
print("FINAL BLOG POST:")
print("="*50)
print(result["draft"])
print(f"\nFinal score: {result['score']}/10")
print(f"Attempts: {result['attempts']}")
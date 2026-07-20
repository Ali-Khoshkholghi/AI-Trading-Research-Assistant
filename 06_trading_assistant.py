from typing import TypedDict, Annotated
import operator
import yfinance as yf
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 1. Tools
# ─────────────────────────────────────────
search = TavilySearch(max_results=5)

def get_stock_price(ticker: str) -> dict:
    """Get real-time stock price from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="5d")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        change = round(current - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2)
        return {
            "ticker": ticker.upper(),
            "price": current,
            "change": change,
            "change_pct": change_pct,
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_risk(entry: float, stop_loss: float, target: float, capital: float, risk_pct: float) -> dict:
    """Calculate position size and risk/reward"""
    risk_per_share = entry - stop_loss
    reward_per_share = target - entry
    rr_ratio = round(reward_per_share / risk_per_share, 2)
    risk_amount = capital * (risk_pct / 100)
    position_size = int(risk_amount / risk_per_share)
    total_cost = round(position_size * entry, 2)
    max_loss = round(position_size * risk_per_share, 2)
    max_gain = round(position_size * reward_per_share, 2)
    return {
        "position_size": position_size,
        "total_cost": total_cost,
        "max_loss": max_loss,
        "max_gain": max_gain,
        "rr_ratio": rr_ratio,
        "risk_amount": risk_amount,
    }

# ─────────────────────────────────────────
# 2. Shared State
# ─────────────────────────────────────────
class TradingState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    intent: str
    price_data: dict
    news_data: str
    risk_data: dict
    final_response: str

# ─────────────────────────────────────────
# 3. LLM
# ─────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ─────────────────────────────────────────
# 4. Agent 1 — Router
# ─────────────────────────────────────────
def router(state: TradingState):
    print("\n🔀 Router: analyzing your query...")
    response = llm.invoke([
        SystemMessage(content="""You are a trading assistant router.
Classify the user query into one of these intents:
- 'price'     → user wants stock/crypto price
- 'news'      → user wants news or sentiment
- 'risk'      → user wants risk or position sizing calculation
- 'both'      → user wants price AND news
- 'general'   → general trading question

Reply with ONLY one word from the list above."""),
        HumanMessage(content=state["query"])
    ])
    intent = response.content.strip().lower()
    print(f"   Intent: {intent}")
    return {"intent": intent}

# ─────────────────────────────────────────
# 5. Agent 2 — Price Agent
# ─────────────────────────────────────────
def price_agent(state: TradingState):
    print("\n📈 Price Agent: fetching real-time data...")
    # Extract ticker from query
    response = llm.invoke([
        SystemMessage(content="Extract the stock or crypto ticker symbol from the query. Reply with ONLY the ticker symbol in uppercase. Examples: NVDA, BTC-USD, AAPL, ETH-USD"),
        HumanMessage(content=state["query"])
    ])
    ticker = response.content.strip()
    print(f"   Ticker: {ticker}")
    data = get_stock_price(ticker)
    print(f"   Price: ${data.get('price', 'N/A')}")
    return {"price_data": data}

# ─────────────────────────────────────────
# 6. Agent 3 — News & Sentiment Agent
# ─────────────────────────────────────────
def news_agent(state: TradingState):
    print("\n📰 News Agent: searching latest news...")
    results = search.invoke(state["query"] + " stock market news today")
    return {"news_data": str(results)}

# ─────────────────────────────────────────
# 7. Agent 4 — Risk Agent
# ─────────────────────────────────────────
def risk_agent(state: TradingState):
    print("\n🧮 Risk Agent: calculating position...")
    response = llm.invoke([
        SystemMessage(content="""Extract trading parameters from the query and return ONLY a JSON object with these keys:
entry, stop_loss, target, capital, risk_pct
If any value is missing use these defaults: capital=10000, risk_pct=2
Example: {"entry": 950, "stop_loss": 920, "target": 1000, "capital": 10000, "risk_pct": 2}"""),
        HumanMessage(content=state["query"])
    ])
    try:
        import json
        params = json.loads(response.content.strip())
        data = calculate_risk(**params)
    except:
        data = {"error": "Could not parse trading parameters. Please provide entry, stop loss, and target prices."}
    return {"risk_data": data}

# ─────────────────────────────────────────
# 8. Agent 5 — Summarizer
# ─────────────────────────────────────────
def summarizer(state: TradingState):
    print("\n📝 Summarizer: preparing response...")

    # Build context from current state AND message history
    context = f"User query: {state['query']}\n"
    
    # Add previous conversation history
    if state.get("messages"):
        context += "\nPrevious conversation:\n"
        for msg in state["messages"]:
            if hasattr(msg, "content"):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                context += f"{role}: {msg.content}\n"

    if state.get("price_data") and state["price_data"]:
        context += f"\nReal-time price data: {state['price_data']}"
    if state.get("news_data") and state["news_data"]:
        context += f"\nLatest news: {state['news_data']}"
    if state.get("risk_data") and state["risk_data"]:
        context += f"\nRisk calculation: {state['risk_data']}"

    # Warn if no real data available
    if not any([state.get("price_data"), state.get("news_data"), state.get("risk_data")]):
        context += "\nNote: No fresh data fetched for this query. Answer based on conversation history only. If you don't have enough context, ask the user to be more specific."

    response = llm.invoke([
        SystemMessage(content="""You are a professional trading assistant.
Answer based ONLY on the real data and conversation history provided.
NEVER invent prices, percentages, or statistics.
If you don't have the data, say: 'I don't have that data yet — ask me: What is the price of NVDA?'
Format numbers nicely. Be direct and actionable.
Always end with: '⚠️ This is not financial advice.'"""),
        HumanMessage(content=context)
    ])
    return {
        "final_response": response.content,
        "messages": state.get("messages", []) + [
            HumanMessage(content=state["query"]),
            response
        ]
    }

# ─────────────────────────────────────────
# 9. Routing Logic
# ─────────────────────────────────────────
def route_intent(state: TradingState):
    intent = state.get("intent", "general")
    if intent == "price":
        return "price"
    elif intent == "news":
        return "news"
    elif intent == "risk":
        return "risk"
    elif intent == "both":
        return "price"  # price first, then news
    else:
        return "summarize"

def after_price(state: TradingState):
    if state.get("intent") == "both":
        return "news"
    return "summarize"

# ─────────────────────────────────────────
# 10. Build Graph
# ─────────────────────────────────────────
graph = StateGraph(TradingState)
graph.add_node("router", router)
graph.add_node("price_agent", price_agent)
graph.add_node("news_agent", news_agent)
graph.add_node("risk_agent", risk_agent)
graph.add_node("summarizer", summarizer)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    route_intent,
    {
        "price": "price_agent",
        "news": "news_agent",
        "risk": "risk_agent",
        "summarize": "summarizer"
    }
)
graph.add_conditional_edges(
    "price_agent",
    after_price,
    {"news": "news_agent", "summarize": "summarizer"}
)
graph.add_edge("news_agent", "summarizer")
graph.add_edge("risk_agent", "summarizer")
graph.add_edge("summarizer", END)

# ─────────────────────────────────────────
# 11. Memory (SQLite — persists across sessions!)
# ─────────────────────────────────────────
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("trading_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)

# ─────────────────────────────────────────
# 12. Chat Loop
# ─────────────────────────────────────────
print("="*50)
print("🤖 AI Trading Research Assistant")
print("="*50)
print("Commands: 'quit' to exit")
print("Examples:")
print("  - What is the price of NVDA?")
print("  - Latest news on Bitcoin")
print("  - I want to buy AAPL at 190, stop loss 185, target 200, capital 10000")
print("="*50)

session_id = input("\nEnter your name for memory: ")
config = {"configurable": {"thread_id": f"trader_{session_id}"}}

while True:
    query = input("\nYou: ").strip()
    if query.lower() == "quit":
        print("Goodbye! Your session has been saved.")
        break
    if not query:
        continue

    result = app.invoke({
        "messages": [],
        "query": query,
        "intent": "",
        "price_data": {},
        "news_data": "",
        "risk_data": {},
        "final_response": ""
    }, config=config)

    print(f"\n🤖 Assistant:\n{result['final_response']}")
    print("-"*50)
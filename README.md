# AI Trading Research Assistant

A multi-agent trading research assistant built with **LangGraph**. It routes a user's natural-language question to the right specialist agent — live stock/crypto prices, market news, or position-sizing/risk math — and answers using only real fetched data, never invented numbers. Conversations persist across sessions via a SQLite checkpointer.

> ⚠️ This project is for educational purposes only. It is **not financial advice**.

## Overview

The assistant is a `StateGraph` of five cooperating nodes:

```
                         ┌────────────┐
                         │   Router   │  (classifies intent via LLM)
                         └─────┬──────┘
              ┌────────────────┼────────────────┬──────────────┐
              ▼                ▼                 ▼              ▼
      ┌───────────────┐ ┌─────────────┐  ┌──────────────┐   (general)
      │  Price Agent  │ │  News Agent │  │  Risk Agent  │       │
      │ (yfinance)    │ │  (Tavily)   │  │ (position &  │       │
      └───────┬───────┘ └──────┬──────┘  │  R:R sizing) │       │
              │  (if "both")   │         └──────┬───────┘       │
              └───────►────────┘                │               │
                       ▼                         ▼               ▼
                  ┌─────────────────────────────────────────────┐
                  │                Summarizer                    │
                  │  (grounds the final answer in fetched data    │
                  │   + prior conversation history)               │
                  └─────────────────────┬───────────────────────┘
                                         ▼
                                        END
```

- **Router** — an LLM call that classifies the query as `price`, `news`, `risk`, `both`, or `general`.
- **Price Agent** — extracts a ticker (stock or crypto) from the query and pulls live price, change %, volume, market cap, and 52-week range from Yahoo Finance (`yfinance`).
- **News Agent** — runs a real-time web search via Tavily for the latest news/sentiment on the topic.
- **Risk Agent** — extracts entry, stop-loss, target, capital, and risk % from the query and calculates position size, max loss/gain, and risk:reward ratio.
- **Summarizer** — combines whatever data was gathered plus prior conversation history into a single grounded, non-hallucinated answer, and always appends a "not financial advice" disclaimer.

Conversation state is checkpointed to `trading_memory.db` (SQLite) keyed by a `thread_id`, so the assistant remembers context across turns and across restarts.

## Project structure

| File | Description |
|---|---|
| `06_trading_assistant.py` | **Main entry point.** The full multi-agent trading assistant with SQLite-backed memory. |
| `first_graph.py` | Foundational LangGraph example — a simple loop where an LLM writes a poem until a checker node detects "DONE". |
| `02_tools.py` | Introduces tool calling — an LLM-driven calculator agent using `@tool` and `bind_tools()`. |
| `03_memory.py` | Introduces checkpointing with `MemorySaver` and `thread_id`-based conversation memory. |
| `04_human_loop.py` | Human-in-the-loop pattern — an email-drafting agent that pauses for approval via `input()`. |
| `05_multi_agent.py` | A 3-agent blog-writing team (Researcher → Writer → Reviewer) with a quality-gated rewrite loop and real Tavily web search. |
| `Docs/langgraph_learning_summary.md` | Notes on the LangGraph concepts learned while building these graphs. |

Files `first_graph.py` through `05_multi_agent.py` are the incremental learning steps that led up to the final assistant in `06_trading_assistant.py`.

## Setup

**Requirements:** Python 3.12+

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/Ali-Khoshkholghi/AI-Trading-Research-Assistant.git
   cd AI-Trading-Research-Assistant
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
   - Get a free Groq API key at [console.groq.com](https://console.groq.com).
   - Get a free Tavily API key at [tavily.com](https://tavily.com).

## Usage

Run the assistant:

```bash
python 06_trading_assistant.py
```

You'll be asked for a name (used as your memory `thread_id`), then you can chat freely:

```
Enter your name for memory: ali

You: What is the price of NVDA?
You: Latest news on Bitcoin
You: I want to buy AAPL at 190, stop loss 185, target 200, capital 10000
You: quit
```

Type `quit` at any time to exit — your conversation is saved and will be restored the next time you run the script with the same name.

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration as a state graph
- [LangChain](https://github.com/langchain-ai/langchain) + [Groq](https://groq.com) (`llama-3.3-70b-versatile`) — LLM reasoning and routing
- [Tavily](https://tavily.com) — real-time web search for news/sentiment
- [yfinance](https://github.com/ranaroussi/yfinance) — live stock/crypto price data from Yahoo Finance
- SQLite (`langgraph-checkpoint-sqlite`) — persistent cross-session memory

# LangGraph Learning Summary
**Date:** May 19, 2026 | **Creator:** Ali Khoshkholghi

---

## Environment Setup
Started from scratch on a managed Mac (no admin access). Installed **pyenv** without sudo to upgrade from Python 3.9.6 to **Python 3.12**. Created a virtual environment (`venv`), learned why isolation matters across projects, and installed core packages: `langgraph`, `langchain-groq`, and `python-dotenv`. Configured a free **Groq API key** (llama-3.3-70b-versatile) as the LLM provider.

---

## Core Concepts Learned
- **State** — a shared TypedDict "notebook" all nodes read from and write to
- **Nodes** — plain Python functions that take state, do work, and return updates
- **Edges** — direct (A→B always) and conditional (A→B or C based on logic)
- **Graph** — a `StateGraph` that connects nodes via edges, compiled with `.compile()`

---

## Graphs Built

### 1. `01_first_graph.py` — The Foundation
A simple loop where an LLM writes a poem and a checker node looks for "DONE" to stop.
```
START → call_llm → check_response → (DONE?) → END
                        ↑________________| (loop if not done)
```

### 2. `02_tools.py` — Adding Tools
Gave the agent a calculator tool. The LLM decides when to use it, gets the result, then forms a final answer.
```
START → call_llm → (tool needed?) → call_tool → call_llm → END
```
**Key concept:** `@tool` decorator + `bind_tools()` + `ToolMessage`

### 3. `03_memory.py` — Checkpointing
Added `MemorySaver` so the agent remembers across multiple `invoke()` calls using a `thread_id`. Tested with a 2-turn conversation where the agent recalled the user's name.
```
invoke 1 → call_llm → END  (saved to thread_id)
invoke 2 → call_llm → END  (loaded + remembered)
```

### 4. `04_human_loop.py` — Human-in-the-Loop
Built an email drafting agent that pauses for human approval before sending using Python's `input()`.
```
START → write_draft → human_review → (yes?) → send_email → END
                                    → (no?)  → reject_email → END
```

### 5. `05_multi_agent.py` — Multi-Agent System
Built a 3-agent blog writing team with real web search via **Tavily**. Includes a quality loop — if the reviewer scores below 7, the writer rewrites (max 3 attempts).
```
START → Researcher (Tavily web search)
           → Writer (drafts blog post)
              → Reviewer (scores 1-10)
                 → (score ≥ 7?) → END
                 → (score < 7?) → Writer (rewrite loop)
```

---

## Key Takeaways
| Concept | What it means |
|---|---|
| `invoke()` | Run the graph once — one full trip through all nodes |
| `thread_id` | Conversation ID — same ID = shared memory |
| `@tool` | Turns a Python function into something the LLM can call |
| `MemorySaver` | Stores state in RAM (lost on exit) |
| `input()` | Pauses the graph for human decision |
| Multi-agent | Each node is a specialist with its own system prompt |

---

*Built entirely in the terminal using Python 3.12, LangGraph 1.2.0, and Groq (free tier).*

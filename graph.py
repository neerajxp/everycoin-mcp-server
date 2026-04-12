"""
LangGraph multi-agent graph for EveryCoin.

Nodes:
  memory_loader     — load short-term session history + long-term user profile
  router            — classify query, decide which agents to run
  market_analyst    — price, gas, market data
  defi_researcher   — protocol TVL, category, chains
  wallet_forensics  — on-chain wallet analysis
  knowledge_expert  — RAG semantic search
  strategist        — synthesize all results + personalize to user profile
  memory_writer     — persist updated session + profile

State flows through every node as AgentState.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import mcp_tools

log = logging.getLogger("everycoin.graph")

# ── LLM ───────────────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        max_tokens=1024,
    )


# ── Memory store (short-term: in-memory, long-term: JSON files) ───────────────

_session_store: dict[str, list[dict]] = {}  # session_id → messages
PROFILES_DIR = Path(__file__).parent / "user_profiles"
MAX_HISTORY = 20  # messages to retain per session


def _load_profile(user_id: str) -> dict:
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "risk_profile": "moderate",
        "tokens_mentioned": [],
        "wallets_watched": [],
        "preferred_chains": [],
    }


def _save_profile(user_id: str, profile: dict) -> None:
    PROFILES_DIR.mkdir(exist_ok=True)
    path = PROFILES_DIR / f"{user_id}.json"
    path.write_text(json.dumps(profile, indent=2))


def _extract_tokens(text: str) -> list[str]:
    """Extract CoinGecko coin IDs from text (ticker or full name)."""
    # Maps both ticker and common name → CoinGecko ID
    known = {
        "BTC": "bitcoin", "BITCOIN": "bitcoin",
        "ETH": "ethereum", "ETHEREUM": "ethereum",
        "SOL": "solana", "SOLANA": "solana",
        "AVAX": "avalanche-2", "AVALANCHE": "avalanche-2",
        "ARB": "arbitrum", "ARBITRUM": "arbitrum",
        "OP": "optimism", "OPTIMISM": "optimism",
        "MATIC": "matic-network", "POLYGON": "matic-network",
        "LINK": "chainlink", "CHAINLINK": "chainlink",
        "AAVE": "aave",
        "UNI": "uniswap", "UNISWAP": "uniswap",
        "CRV": "curve-dao-token", "CURVE": "curve-dao-token",
        "GMX": "gmx",
        "LDO": "lido-dao", "LIDO": "lido-dao",
        "MKR": "maker", "MAKER": "maker",
        "COMP": "compound-governance-token", "COMPOUND": "compound-governance-token",
    }
    upper = text.upper()
    found = {coin_id for keyword, coin_id in known.items() if keyword in upper}
    return list(found)


def _extract_wallets(text: str) -> list[str]:
    return re.findall(r"0x[a-fA-F0-9]{40}", text)


# ── AgentState ────────────────────────────────────────────────────────────────

def _merge_tool_results(a: dict, b: dict) -> dict:
    """Reducer: merge concurrent agent writes into tool_results."""
    return {**a, **b}


class AgentState(TypedDict):
    # conversation
    messages: Annotated[list, add_messages]   # short-term: full chat history
    session_id: str
    user_id: str
    user_query: str

    # memory
    session_history: list[dict]               # loaded from _session_store
    user_profile: dict                        # loaded from user_profiles/

    # routing
    query_type: str                           # set by router node
    active_agents: list[str]                  # which nodes to run

    # inter-node results — Annotated reducer handles parallel agent writes
    tool_results: Annotated[dict[str, Any], _merge_tool_results]

    # output
    final_answer: str


# ── Node: memory_loader ───────────────────────────────────────────────────────

def memory_loader(state: AgentState) -> dict:
    session_id = state["session_id"]
    user_id = state.get("user_id", "anonymous")

    history = _session_store.get(session_id, [])
    profile = _load_profile(user_id)

    log.info("[memory_loader] session=%s history=%d msgs profile=%s",
             session_id, len(history), profile.get("risk_profile"))

    return {
        "session_history": history,
        "user_profile": profile,
    }


# ── Node: router ──────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a query router for a crypto AI assistant.
Classify the user query into one of these types and decide which agents are needed.

Query types and their agents:
- "price"   → ["market_analyst"]
- "defi"    → ["defi_researcher", "knowledge_expert"]
- "wallet"  → ["wallet_forensics"]
- "explain" → ["knowledge_expert"]
- "strategy"→ ["market_analyst", "knowledge_expert"]
- "complex" → ["market_analyst", "defi_researcher", "knowledge_expert"]

Respond with ONLY a JSON object:
{"query_type": "<type>", "active_agents": ["agent1", "agent2"]}
"""


def router(state: AgentState) -> dict:
    query = state["user_query"]
    history = state.get("session_history", [])

    # Build context from recent history
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        history_text = f"\nRecent conversation:\n{history_text}\n"

    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=f"{history_text}Current query: {query}"),
    ])

    try:
        raw = response.content
        # extract JSON even if wrapped in markdown
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        query_type = parsed.get("query_type", "complex")
        active_agents = parsed.get("active_agents", ["knowledge_expert"])
    except Exception:
        query_type = "complex"
        active_agents = ["market_analyst", "knowledge_expert"]

    # always add strategist — it synthesizes everything
    if "strategist" not in active_agents:
        active_agents.append("strategist")

    log.info("  🧭 ROUTER → type: %-10s  agents: %s", query_type, active_agents)
    return {"query_type": query_type, "active_agents": active_agents}


# ── Node: market_analyst ──────────────────────────────────────────────────────

_MARKET_SYSTEM = """You are a crypto market analyst.
Use the provided tool results to produce a concise factual market summary.
Focus on: price, trend, gas costs, market context.
Be data-driven. 3-5 sentences max."""


async def market_analyst(state: AgentState) -> dict:
    query = state["user_query"]
    tool_results = dict(state.get("tool_results", {}))

    # Detect tokens in query — returns CoinGecko IDs directly
    tokens = _extract_tokens(query)
    if not tokens:
        tokens = ["ethereum"]  # default to ETH if none found

    gathered = {}
    for token in tokens[:2]:  # max 2 tokens to keep latency reasonable
        price_data = await mcp_tools.get_token_price(token)
        gathered[f"price_{token}"] = price_data

    gas_data = await mcp_tools.get_gas_price()
    gathered["gas"] = gas_data

    # Summarize with LLM
    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_MARKET_SYSTEM),
        HumanMessage(content=f"Query: {query}\nTool data: {json.dumps(gathered)}"),
    ])

    tool_results["market_analyst"] = {
        "summary": response.content,
        "raw_data": gathered,
    }
    log.info("  📈 MARKET ANALYST done | tokens fetched: %s", list(gathered.keys()))
    return {"tool_results": tool_results}


# ── Node: defi_researcher ─────────────────────────────────────────────────────

_DEFI_SYSTEM = """You are a DeFi protocol researcher.
Use the provided tool results to give a factual protocol overview.
Focus on: TVL, category, chains, key characteristics.
Be concise. 3-5 sentences max."""


async def defi_researcher(state: AgentState) -> dict:
    query = state["user_query"]
    tool_results = dict(state.get("tool_results", {}))

    # Detect protocol names in query
    known_protocols = [
        "aave", "uniswap", "curve", "lido", "gmx", "pendle",
        "compound", "maker", "balancer", "convex", "frax",
    ]
    mentioned = [p for p in known_protocols if p in query.lower()]
    if not mentioned:
        mentioned = ["aave"]  # default

    gathered = {}
    for protocol in mentioned[:2]:
        stats = await mcp_tools.get_defi_stats(protocol)
        gathered[protocol] = stats

    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_DEFI_SYSTEM),
        HumanMessage(content=f"Query: {query}\nProtocol data: {json.dumps(gathered)}"),
    ])

    tool_results["defi_researcher"] = {
        "summary": response.content,
        "raw_data": gathered,
    }
    log.info("  🏦 DEFI RESEARCHER done | protocols fetched: %s", list(gathered.keys()))
    return {"tool_results": tool_results}


# ── Node: wallet_forensics ────────────────────────────────────────────────────

_WALLET_SYSTEM = """You are a crypto wallet forensics analyst.
Analyze the wallet data and identify: balance level, activity pattern, any notable patterns.
Flag any concerns (e.g. very recent activity, large balance, unusual tx patterns).
Be concise. 3-5 sentences max."""


async def wallet_forensics(state: AgentState) -> dict:
    query = state["user_query"]
    tool_results = dict(state.get("tool_results", {}))

    wallets = _extract_wallets(query)
    if not wallets:
        tool_results["wallet_forensics"] = {
            "summary": "No wallet address found in the query.",
            "raw_data": {},
        }
        return {"tool_results": tool_results}

    gathered = {}
    for addr in wallets[:1]:  # analyze first wallet only
        data = await mcp_tools.analyze_wallet(addr)
        gathered[addr] = data

    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_WALLET_SYSTEM),
        HumanMessage(content=f"Query: {query}\nWallet data: {json.dumps(gathered)}"),
    ])

    tool_results["wallet_forensics"] = {
        "summary": response.content,
        "raw_data": gathered,
    }
    log.info("  🔍 WALLET FORENSICS done | wallets: %s", list(gathered.keys()))
    return {"tool_results": tool_results}


# ── Node: knowledge_expert ────────────────────────────────────────────────────

_KNOWLEDGE_SYSTEM = """You are a crypto knowledge expert.
Use the retrieved knowledge chunks to provide accurate background context.
Cite the key concepts that are relevant to the user's question.
Be concise. 3-5 sentences max."""

# map query_type to rag topic
_TOPIC_MAP = {
    "defi": "defi",
    "explain": None,      # search all topics
    "strategy": "strategy",
    "complex": None,
}


def knowledge_expert(state: AgentState) -> dict:
    query = state["user_query"]
    query_type = state.get("query_type", "complex")
    tool_results = dict(state.get("tool_results", {}))

    topic = _TOPIC_MAP.get(query_type)

    # detect topic from query keywords
    if "security" in query.lower() or "rug" in query.lower() or "scam" in query.lower():
        topic = "security"
    elif "l2" in query.lower() or "arbitrum" in query.lower() or "rollup" in query.lower():
        topic = "l2"

    rag_result = mcp_tools.search_knowledge(query, topic=topic)
    chunks = rag_result.get("results", [])

    if not chunks:
        tool_results["knowledge_expert"] = {
            "summary": "No relevant knowledge found for this query.",
            "sources": [],
        }
        return {"tool_results": tool_results}

    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)

    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_KNOWLEDGE_SYSTEM),
        HumanMessage(content=f"Query: {query}\n\nRetrieved knowledge:\n{context}"),
    ])

    tool_results["knowledge_expert"] = {
        "summary": response.content,
        "sources": [c["source"] for c in chunks],
    }
    log.info("  📚 KNOWLEDGE EXPERT done | RAG sources: %s", [c["source"] for c in chunks])
    return {"tool_results": tool_results}


# ── Node: strategist ──────────────────────────────────────────────────────────

_STRATEGIST_SYSTEM = """You are the EveryCoin Strategist — a senior crypto advisor.
You receive analysis from specialist agents and synthesize it into a final personalized answer.

Your output must:
- Lead with the key insight
- Integrate data from all available agent summaries
- Personalize to the user's risk profile
- Use **bold** for key terms, tickers, protocols
- Use bullet points for lists
- Prefix warnings with ⚠
- End with DYOR reminder on high-stakes advice
- Be concise — no padding, no filler"""


def strategist(state: AgentState) -> dict:
    query = state["user_query"]
    tool_results = state.get("tool_results", {})
    user_profile = state.get("user_profile", {})
    session_history = state.get("session_history", [])

    # Build context from all agent summaries
    summaries = []
    for agent_name, result in tool_results.items():
        if isinstance(result, dict) and "summary" in result:
            summaries.append(f"[{agent_name}]\n{result['summary']}")

    agents_context = "\n\n".join(summaries) if summaries else "No agent data available."

    # Recent conversation context
    history_text = ""
    if session_history:
        recent = session_history[-6:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    profile_text = (
        f"User risk profile: {user_profile.get('risk_profile', 'moderate')}\n"
        f"Previously mentioned tokens: {', '.join(user_profile.get('tokens_mentioned', [])) or 'none'}\n"
        f"Watched wallets: {len(user_profile.get('wallets_watched', []))} saved"
    )

    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_STRATEGIST_SYSTEM),
        HumanMessage(content=(
            f"User query: {query}\n\n"
            f"User profile:\n{profile_text}\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Agent findings:\n{agents_context}"
        )),
    ])

    log.info("  🧠 STRATEGIST done | synthesized from: %s", list(tool_results.keys()))
    return {"final_answer": response.content}


# ── Node: memory_writer ───────────────────────────────────────────────────────

def memory_writer(state: AgentState) -> dict:
    session_id = state["session_id"]
    user_id = state.get("user_id", "anonymous")
    query = state["user_query"]
    answer = state.get("final_answer", "")
    profile = dict(state.get("user_profile", {}))

    # Update short-term session history
    history = list(_session_store.get(session_id, []))
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    # Keep only last MAX_HISTORY messages
    _session_store[session_id] = history[-MAX_HISTORY:]

    # Update long-term profile
    new_tokens = _extract_tokens(query + " " + answer)
    existing_tokens = set(profile.get("tokens_mentioned", []))
    profile["tokens_mentioned"] = list(existing_tokens | set(new_tokens))

    new_wallets = _extract_wallets(query)
    existing_wallets = set(profile.get("wallets_watched", []))
    profile["wallets_watched"] = list(existing_wallets | set(new_wallets))

    if user_id != "anonymous":
        _save_profile(user_id, profile)

    log.info("  💾 MEMORY WRITER | session: %d msgs | profile: %s | tokens: %s",
             len(_session_store[session_id]), user_id,
             profile.get("tokens_mentioned", []))
    return {}


# ── Conditional routing ───────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> list[str]:
    """Return which agent nodes to run based on router decision."""
    active = state.get("active_agents", [])
    # remove strategist — it runs after agents via fixed edge
    return [a for a in active if a != "strategist"]


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("memory_loader", memory_loader)
    builder.add_node("router", router)
    builder.add_node("market_analyst", market_analyst)
    builder.add_node("defi_researcher", defi_researcher)
    builder.add_node("wallet_forensics", wallet_forensics)
    builder.add_node("knowledge_expert", knowledge_expert)
    builder.add_node("strategist", strategist)
    builder.add_node("memory_writer", memory_writer)

    # Fixed edges
    builder.add_edge(START, "memory_loader")
    builder.add_edge("memory_loader", "router")

    # Conditional fan-out from router → active agent nodes
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "market_analyst": "market_analyst",
            "defi_researcher": "defi_researcher",
            "wallet_forensics": "wallet_forensics",
            "knowledge_expert": "knowledge_expert",
        },
    )

    # All agent nodes → strategist
    for agent in ["market_analyst", "defi_researcher", "wallet_forensics", "knowledge_expert"]:
        builder.add_edge(agent, "strategist")

    # Strategist → memory_writer → END
    builder.add_edge("strategist", "memory_writer")
    builder.add_edge("memory_writer", END)

    return builder.compile()


# Singleton graph instance
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public entry point ────────────────────────────────────────────────────────

async def run_graph(
    user_query: str,
    messages: list[dict],
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> str:
    """
    Run the LangGraph agent pipeline.
    Returns the final synthesized answer.
    """
    session_id = session_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "session_id": session_id,
        "user_id": user_id,
        "user_query": user_query,
        "session_history": [],
        "user_profile": {},
        "query_type": "",
        "active_agents": [],
        "tool_results": {},
        "final_answer": "",
    }

    graph = get_graph()

    log.info("━" * 60)
    log.info("▶ NEW REQUEST  session=%s  user=%s", session_id, user_id)
    log.info("  Query: %s", user_query)
    log.info("━" * 60)

    result = await graph.ainvoke(initial_state)

    agents_used = list(result.get("tool_results", {}).keys())
    log.info("━" * 60)
    log.info("✓ COMPLETED")
    log.info("  Agents used : %s", agents_used)
    log.info("  Query type  : %s", result.get("query_type", "?"))
    log.info("━" * 60)

    return result.get("final_answer", "No response generated.")

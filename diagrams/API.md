# EveryCoin AI — System Architecture

> Technical overview for architects and system users.  
> Stack: **Next.js 15 (Static Export)** · **Python / Starlette** · **LangGraph** · **XGBoost MLOps** · **ChromaDB RAG** · **Railway**

---

## High-Level System Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER (Browser / Mobile)                            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ HTTPS
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NEXT.JS FRONTEND  (Static Export / CDN)                 │
│                                                                             │
│  Pages              Components                     Widgets                  │
│  ─────────────      ──────────────────────────     ────────────────────     │
│  / (Signals)   →   TotalAIScore (FOMO Meter)       PolymarketWidget         │
│  /portfolio    →   BTCForecast                     ManifoldWidget           │
│  /agent        →   AssetsTable                                              │
│  /analytics    →   WhaleWatcher              Layout                         │
│  /settings     →   AIRecommendations         ──────────────────             │
│                    AgentChat + AgentFAB      Headerbar                      │
│                    WalletConnector           LeftSidebar (desktop)          │
│                    PriceTicker               BottomMobileNav (mobile)       │
│                                              NewsSidebarWrapper             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ REST API (HTTPS)
                                │ Railway hosted · CORS enabled
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PYTHON BACKEND  (Starlette / Uvicorn)                     │
│                          server.py — API Gateway                            │
│                                                                             │
│  Route                       Handler               Description              │
│  ──────────────────────────  ──────────────────    ─────────────────────    │
│  POST /api/chat              handle_chat           LangGraph AI agent       │
│  POST /mcp                   handle_mcp            MCP JSON-RPC tools       │
│  GET  /prices                handle_prices         CoinGecko price proxy    │
│  GET  /predict/ai-score      handle_predict        XGBoost ML scores        │
│  GET  /predict/btc-momentum  handle_btc_momentum   BTC momentum signals     │
│  GET  /predict/price-target  handle_price_target   Price target forecast    │
│  GET  /predict/btc-journey   handle_btc_journey    BTC journey data         │
│  GET  /predict/price-history handle_price_history  Historical price data    │
│  GET  /whale/signals         handle_whale_signals  On-chain whale tracker   │
│  GET  /predict/polymarket    handle_polymarket      Polymarket crowd bets    │
│  GET  /predict/manifold      handle_manifold        Manifold prediction mkt  │
│  GET  /predict/trending-chips handle_trending_chips CoinGecko trending      │
│  GET  /predict/market-narrative handle_market_narrative Claude Haiku brief  │
│  GET  /health                handle_health         Health check             │
└──────┬──────────┬────────────┬──────────────┬──────────────────────────────┘
       │          │            │              │
       ▼          ▼            ▼              ▼
```

---

## Backend Sub-Systems

### 1. LangGraph Multi-Agent Pipeline (`graph.py`)

```
User Message
     │
     ▼
┌────────────────┐
│ memory_loader  │  Load session history + user profile
└───────┬────────┘
        ▼
┌────────────────┐
│    router      │  Classify query → decide agent path   [Claude Haiku]
└───────┬────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌───────────────────┐            ┌──────────────────────┐
│  market_analyst   │            │   defi_researcher    │
│  Price · Gas      │            │   TVL · Protocols    │
│  [Claude Haiku]   │            │   [Claude Haiku]     │
└─────────┬─────────┘            └──────────┬───────────┘
          │                                 │
          ▼                                 ▼
┌───────────────────┐            ┌──────────────────────┐
│ wallet_forensics  │            │  knowledge_expert    │
│ On-chain analysis │            │  RAG semantic search │
│ [Claude Haiku]    │            │  [ChromaDB + Haiku]  │
└─────────┬─────────┘            └──────────┬───────────┘
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
               ┌──────────────────┐
               │   strategist     │  Synthesize all results
               │  [Claude Sonnet] │  Personalized response
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │  memory_writer   │  Persist session + profile
               └────────┬─────────┘
                        ▼
                   Response to User
```

### 2. RAG Knowledge Engine (`rag.py`)

```
Knowledge Files (*.md)
  ├── defi_protocols.md
  ├── market_strategy.md
  ├── security.md
  └── l2_scaling.md
        │
        ▼ chunk + embed (BAAI/bge-small-en-v1.5 via fastembed)
        ▼
  ┌─────────────────┐
  │   ChromaDB      │  Persistent vector store
  │  (chroma_db/)   │
  └────────┬────────┘
           │ semantic similarity search
           ▼
     knowledge_expert node
```

### 3. MLOps Pipeline (`mlops/`)

```
                  ┌──────────────────────────────────────────┐
                  │           MLOps Pipeline                  │
                  │                                           │
  CoinGecko API → │ fetch.py   → Raw OHLCV + market data     │
                  │ features.py → RSI · MACD · ATR signals   │
                  │ db.py      → SQLite feature store        │
                  │ train.py   → XGBoost classifier          │
                  │ promote.py → Best model → HuggingFace Hub│
                  │ serve.py   → Load model → AI Score 0-100 │
                  │ scheduler.py → Cron: retrain + backfill  │
                  └──────────────────┬───────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   AI Score Output    │
                          │  0-44  → SELL        │
                          │  45-74 → HOLD        │
                          │  75-100 → BUY        │
                          └─────────────────────┘
```

### 4. MCP Tools (`mcp_tools.py`)

```
  MCP JSON-RPC (/mcp)
       │
       ├── get_price          CoinGecko live price
       ├── get_portfolio      Portfolio value calc
       ├── get_whale_activity Whale wallet tracker
       ├── get_defi_tvl       DeFi protocol TVL
       └── get_gas_price      ETH gas oracle
```

### 5. External Data Sources

```
  ┌─────────────────────────────────────────────────────┐
  │               External APIs                          │
  │                                                      │
  │  CoinGecko API    → Prices · Trending · Market data  │
  │  Polymarket API   → Crypto prediction markets        │
  │  Manifold API     → Community prediction markets     │
  │  Etherscan API    → Wallet · Token transactions      │
  │  Ankr RPC         → BSC multichain balances          │
  │  Solana RPC       → SOL + SPL token balances         │
  │  HuggingFace Hub  → Model registry (XGBoost)         │
  │  Anthropic API    → Claude Haiku + Sonnet            │
  └─────────────────────────────────────────────────────┘
```

---

## Frontend Feature Map

### Signals Page (`/`)

```
  PriceTicker          Live scrolling price strip (BTC · ETH · ADA · SOL)
       │
  TotalAIScore         FOMO Meter — blended ML + live signal score
       │
  BTCForecast          Price target · direction · confidence · window
       │
  WhaleWatcher         On-chain large wallet movements
       │
  AssetsTable          Portfolio table with AI BUY/HOLD/SELL per coin
       │
  AIRecommendations    Claude Haiku market narrative with signal highlights
       │
  PolymarketWidget     Crowd prediction markets (mobile only — sidebar on desktop)
  ManifoldWidget       Community prediction markets
```

### Portfolio Page (`/portfolio`)

```
  WalletConnector
       ├── MetaMask (ETH/EVM)
       ├── Phantom (Solana)
       └── Manual address input
            │
            ▼
  Multi-chain asset fetch
       ├── Ethereum (Etherscan v2)
       ├── BSC (Ankr RPC)
       └── Solana (RPC + SPL tokens)
            │
            ▼
  CoinGecko price enrichment
            │
            ▼
  AI Signal per asset (BUY / HOLD / SELL)
            │
            ▼
  TotalWorthHero + AssetsTable view
```

### Agent Page (`/agent`)

```
  Trending Chips          CoinGecko trending → dynamic chip prompts
  (Row 1: static scroll · Row 2: marquee auto-scroll)
       │
  Search Box              Text + Voice input
       │
  AgentChat               LangGraph multi-agent pipeline
       │
  AgentFAB                Floating button (desktop overlay panel)
```

### Desktop Sidebar

```
  LeftSidebar             Signals · Portfolio · Agent · Analytics · Settings
  NewsSidebarWrapper      PolymarketWidget + ManifoldWidget (desktop only)
```

---

## Data Flow — End to End

```
Browser
  │
  │  1. Page load → fetch /prices, /predict/ai-score, /predict/btc-momentum
  │
  ├─→  Python Backend
  │         │
  │         ├── CoinGecko → prices cached in memory (5 min TTL)
  │         ├── MLOps serve.py → XGBoost AI score (cached 5 min)
  │         └── BTC momentum signals (cached 30 min)
  │
  │  2. Sidebar → fetch /predict/polymarket, /predict/manifold
  │
  ├─→  Python Backend
  │         ├── Polymarket Gamma API (cached 1 hr)
  │         └── Manifold API × 5 search terms parallel (cached 1 hr)
  │
  │  3. AIRecommendations → fetch /predict/market-narrative
  │
  ├─→  Python Backend
  │         ├── BTC momentum + whale signals
  │         └── Claude Haiku → narrative (cached 30 min, busted on refresh)
  │
  │  4. Agent chat → POST /api/chat
  │
  ├─→  Python Backend
  │         └── LangGraph pipeline → Haiku agents → Sonnet strategist
  │
  │  5. Portfolio → wallet addresses
  │
  └─→  Etherscan / Ankr / Solana RPC → CoinGecko prices
```

---

## Deployment

```
  ┌───────────────────────────┐     ┌──────────────────────────────┐
  │   GitLab (Next.js repo)   │     │  GitHub (Python server repo) │
  │   Static export → out/    │     │  Procfile: uvicorn server    │
  └───────────────────────────┘     └──────────────┬───────────────┘
                                                   │ Auto-deploy
                                                   ▼
                                          ┌─────────────────┐
                                          │    Railway.app  │
                                          │  Python server  │
                                          │  :8000          │
                                          └─────────────────┘

  Environment Variables
  ├── NEXT_PUBLIC_CHAT_API_URL   → Railway backend URL
  ├── ANTHROPIC_API_KEY          → Claude API
  ├── COINGECKO_API_KEY          → Market data
  ├── ETHERSCAN_API_KEY          → Wallet data
  └── NEXT_PUBLIC_GA_MEASUREMENT_ID → Analytics
```

---

## Caching Strategy

| Endpoint                    | TTL     | Notes                           |
| --------------------------- | ------- | ------------------------------- |
| `/prices`                   | 5 min   | In-memory                       |
| `/predict/ai-score`         | 5 min   | In-memory                       |
| `/predict/btc-momentum`     | 30 min  | In-memory + client localStorage |
| `/predict/market-narrative` | 30 min  | Busted via `?t=` param          |
| `/predict/polymarket`       | 1 hour  | In-memory                       |
| `/predict/manifold`         | 1 hour  | In-memory                       |
| `/predict/trending-chips`   | 30 min  | In-memory                       |
| Frontend prices             | Session | sessionStorage + localStorage   |

---

## Tech Stack Summary

| Layer              | Technology                                                   |
| ------------------ | ------------------------------------------------------------ |
| Frontend           | Next.js 15, React, TypeScript, Tailwind CSS                  |
| Backend            | Python, Starlette, Uvicorn                                   |
| AI Agents          | LangGraph, Claude Haiku (agents), Claude Sonnet (strategist) |
| ML Model           | XGBoost, scikit-learn, pandas, HuggingFace Hub               |
| RAG                | ChromaDB, fastembed (BAAI/bge-small-en-v1.5)                 |
| Blockchain         | Etherscan v2, Ankr RPC, Solana RPC                           |
| Prediction Markets | Polymarket Gamma API, Manifold Markets API                   |
| Market Data        | CoinGecko API                                                |
| Deployment         | Railway (backend), Static CDN (frontend)                     |
| Analytics          | Google Analytics GA4                                         |

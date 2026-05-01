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

## External API Reference

All third-party endpoints the system calls, grouped by provider.

---

### CoinGecko

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://api.coingecko.com/api/v3/simple/price` | `server.py /prices`, `mcp_tools.py`, `mlops/fetch.py`, `priceService.ts` | Live USD prices + 24h change for one or more coins |
| `GET https://api.coingecko.com/api/v3/coins/bitcoin/market_chart` | `server.py /predict/btc-momentum` | BTC OHLCV history for momentum signal calculation |
| `GET https://api.coingecko.com/api/v3/search/trending` | `server.py /predict/trending-chips` | Trending coins list used for dynamic agent prompt chips |

**Auth:** Optional `x-cg-demo-api-key` header (set via `COINGECKO_API_KEY` env var). Free tier applies when key is absent.

---

### Anthropic (Claude)

| Endpoint | Called From | Model | Purpose |
|---|---|---|---|
| `POST https://api.anthropic.com/v1/messages` | `graph.py` — router, market_analyst, defi_researcher, wallet_forensics, knowledge_expert | claude-haiku-4-5 | Lightweight agent nodes — routing, analysis, research |
| `POST https://api.anthropic.com/v1/messages` | `graph.py` — strategist | claude-sonnet-4-6 | Final synthesis and personalized user response |
| `POST https://api.anthropic.com/v1/messages` | `server.py /predict/market-narrative` | claude-haiku-4-5 | Generates the AI Market Brief narrative + signal feed |

**Auth:** `ANTHROPIC_API_KEY` environment variable (required).

---

### OKX

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP` | `server.py /predict/btc-momentum`, `/whale/signals` | BTC perpetual funding rate — indicates leverage bias (longs vs shorts) |
| `GET https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP` | `server.py /whale/signals` | BTC perpetual open interest in USD — trend strength signal |
| `GET https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader?instId=BTC-USDT-SWAP&period=1H` | `server.py /whale/signals` | Top trader long/short ratio — contrarian sentiment signal |

**Auth:** None required (public endpoints).

---

### Mempool.space

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://mempool.space/api/v1/blocks/tip/height` | `server.py /whale/signals` | Latest Bitcoin block height |
| `GET https://mempool.space/api/block-height/{height}` | `server.py /whale/signals` | Block hash for a given height |
| `GET https://mempool.space/api/block/{hash}` | `server.py /whale/signals` | Block metadata (timestamp, tx count) |
| `GET https://mempool.space/api/block/{hash}/txs/0` | `server.py /whale/signals` | Transactions in a block — parsed for whale-size BTC moves |

**Auth:** None required (public API).

---

### Polymarket

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://gamma-api.polymarket.com/markets?active=true&closed=false&tag_slug=crypto` | `server.py /predict/polymarket` | Active crypto prediction markets with YES probability prices |
| `GET https://gamma-api.polymarket.com/markets?active=true&closed=false&tag_slug=bitcoin` | `server.py /predict/polymarket` | Active Bitcoin-tagged markets as supplementary source |

**Auth:** None required (public API). Market URLs constructed as `https://polymarket.com/event/{slug}`.

---

### Manifold Markets

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://api.manifold.markets/v0/search-markets?term={keyword}&filter=open&sort=liquidity&limit=10` | `server.py /predict/manifold` | Search for open community prediction markets by keyword (BTC, ETH, crypto, etc.) — called in parallel for multiple terms |

**Auth:** None required (public API).

---

### Etherscan / BSCScan

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://api.etherscan.io/v2/api` | `next/src/constants/chains.ts`, `mcp_tools.py` | Ethereum wallet token balances, transaction history |
| `GET https://api.bscscan.com/api` | `next/src/constants/chains.ts` | BSC wallet token balances (accepts same Etherscan API key) |
| `GET https://api.etherscan.io/api` | `mcp_tools.py`, `mlops/fetch.py` | Token metadata, historical transactions for wallet forensics |

**Auth:** `ETHERSCAN_API_KEY` environment variable (required for portfolio features).

---

### Solana RPC

| Endpoint | Called From | Purpose |
|---|---|---|
| `POST https://api.mainnet-beta.solana.com` | `next/src/services/solanaService.ts` | SOL balance + SPL token accounts for a wallet address |

**Auth:** None required (public RPC). Rate-limited — upgrade to a private RPC (Helius, QuickNode) for production scale.

---

### BSC (Binance Smart Chain) RPC

| Endpoint | Called From | Purpose |
|---|---|---|
| `POST https://bsc-dataseed.binance.org/` | `next/src/services/bscService.ts` | BEP-20 token balances via `eth_call` JSON-RPC |

**Auth:** None required (public RPC).

---

### DeFiLlama

| Endpoint | Called From | Purpose |
|---|---|---|
| `GET https://api.llama.fi/protocol/{protocol}` | `mcp_tools.py get_defi_tvl`, `mlops/fetch.py` | Protocol TVL (Total Value Locked) used in DeFi analysis and agent tools |

**Auth:** None required (public API).

---

### HuggingFace Hub

| Endpoint | Called From | Purpose |
|---|---|---|
| `https://huggingface.co/{repo}` | `mlops/promote.py`, `mlops/serve.py` | Upload trained XGBoost model after retraining; download best model at server startup |

**Auth:** `HUGGINGFACE_TOKEN` environment variable (required for MLOps pipeline).

---

## External API Cache TTL Summary

| Provider | Endpoint | Server TTL | Notes |
|---|---|---|---|
| CoinGecko | `/prices` | 5 min | In-memory |
| CoinGecko | BTC market chart | 30 min | Part of btc-momentum cache |
| CoinGecko | Trending chips | 30 min | In-memory |
| OKX | Funding rate | 30 min | Bundled in whale/momentum cache |
| OKX | Open interest | 5 min | Part of whale signals cache |
| OKX | Long/short ratio | 5 min | Part of whale signals cache |
| Mempool.space | Block txs | 5 min | Part of whale signals cache |
| Polymarket Gamma | Markets | 15 min | In-memory |
| Manifold | Search markets | 1 hour | In-memory |
| Anthropic | Market narrative | 30 min | Busted via `?t=` param |

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

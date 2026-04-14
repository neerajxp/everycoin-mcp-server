graph TB
    User(["👤 User\nBrowser"])

    subgraph Frontend["Frontend — Next.js on Apache"]
        UI["AgentChat.tsx\nReact Component"]
        Session["SESSION_ID\ncrypto.randomUUID()"]
    end

    subgraph Railway["Backend — Python on Railway"]
        Server["Starlette\nserver.py\nPOST /api/chat\nPOST /mcp\nGET /health"]

        subgraph LangGraph["LangGraph Orchestrator — graph.py"]
            ML["memory_loader\n── loads session history\n── loads user profile"]
            RT["router\n(Claude Haiku)\n── classifies query_type\n── picks active_agents"]

            subgraph Agents["Specialist Agents (Claude Haiku — parallel fan-out)"]
                MA["market_analyst\n── get_token_price\n── get_gas_price"]
                DR["defi_researcher\n── get_defi_stats\n── search_knowledge"]
                WF["wallet_forensics\n── analyze_wallet\n── get_gas_price"]
                KE["knowledge_expert\n── search_knowledge"]
            end

            ST["strategist\n(Claude Sonnet)\n── synthesizes all tool_results\n── writes final_answer"]
            MW["memory_writer\n── saves session to RAM\n── updates user profile JSON"]
        end

        subgraph MCP["MCP Tool Layer — mcp_tools.py"]
            T1["get_token_price"]
            T2["get_defi_stats"]
            T3["get_gas_price"]
            T4["analyze_wallet"]
            T5["search_knowledge"]
        end

        subgraph Memory["Memory"]
            STM["Short-term Memory\nRAM dict\nkeyed by session_id"]
            LTM["Long-term Memory\nuser_profiles/*.json\nkeyed by user_id"]
        end

        subgraph RAG["RAG Engine — rag.py"]
            Embed["fastembed\nBAAI/bge-small-en-v1.5\nONNX — no PyTorch"]
            Chroma["ChromaDB\nchroma_db/\nPersistentClient"]
            KB["Knowledge Base\nknowledge/*.md\ndefi · security · strategy · l2"]
        end
    end

    subgraph ExternalAPIs["External APIs"]
        CG["CoinGecko\nToken prices"]
        DL["DeFiLlama\nDeFi TVL stats"]
        ES["Etherscan\nWallet · Gas"]
        Anthropic["Anthropic API\nHaiku · Sonnet"]
    end

    %% User flow
    User -->|"HTTPS POST /api/chat\n{messages, session_id}"| UI
    UI --> Server

    %% LangGraph flow
    Server --> ML
    ML -->|"reads"| STM
    ML -->|"reads"| LTM
    ML --> RT
    RT -->|"conditional fan-out"| MA & DR & WF & KE
    MA & DR & WF & KE -->|"tool_results merged"| ST
    ST --> MW
    MW -->|"writes"| STM
    MW -->|"writes"| LTM
    MW -->|"final_answer"| Server
    Server -->|"JSON response"| User

    %% Agent → MCP tools
    MA --> T1 & T3
    DR --> T2 & T5
    WF --> T4 & T3
    KE --> T5

    %% MCP → External
    T1 --> CG
    T2 --> DL
    T3 & T4 --> ES
    T5 --> Chroma

    %% RAG internals
    Chroma --> Embed
    KB -->|"chunked on startup\n300 words / 50 overlap"| Embed
    Embed -->|"HNSW cosine index"| Chroma

    %% LLM calls
    RT & MA & DR & WF & KE -->|"Claude Haiku"| Anthropic
    ST -->|"Claude Sonnet"| Anthropic

    %% Styles
    classDef agent fill:#1a1a2e,stroke:#00b894,color:#fff
    classDef tool fill:#16213e,stroke:#0984e3,color:#fff
    classDef storage fill:#0f3460,stroke:#e17055,color:#fff
    classDef external fill:#533483,stroke:#a29bfe,color:#fff
    class MA,DR,WF,KE,ST,RT,ML,MW agent
    class T1,T2,T3,T4,T5 tool
    class STM,LTM,Chroma,KB storage
    class CG,DL,ES,Anthropic external

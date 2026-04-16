# EveryCoin — MLOps Pipeline

End-to-end MLOps pipeline for training and serving AI models that power the EveryCoin portfolio intelligence features (AI Score, price direction signals, agent context).

**Tools:** MLflow · Prefect/ZenML · Evidently AI · Prometheus/Grafana · FastAPI · Docker · GitHub Actions · Railway

```mermaid
flowchart TD
    subgraph INGEST["① Data Ingestion"]
        A1["CoinGecko — Price · Volume · Market Cap"]
        A2["DeFiLlama — TVL · Protocol Stats"]
        A3["Etherscan — Wallet · On-chain Activity"]
        A4["News / Reddit — Sentiment Sources"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph RAWSTORE["② Raw Data Storage"]
        B1["PostgreSQL / CSV / Parquet\ntime-series market data"]
    end

    subgraph FEATURES["③ Feature Engineering"]
        C1["Price — SMA · EMA · RSI · MACD · Bollinger · Returns"]
        C2["Sentiment — News polarity · Reddit mention volume"]
        C3["On-chain — Wallet activity · Gas trends · Whale signals"]
        C1 --> C2 --> C3
    end

    subgraph FSTORE["④ Feature Store"]
        B2["Feast / manual pipeline\nversioned engineered features"]
    end

    subgraph TRAIN["⑤ Model Training"]
        D3["Prefect / ZenML — orchestrated pipeline"]
        D1["XGBoost · LightGBM · LSTM · Random Forest"]
        D2["MLflow — metrics · params · artifacts"]
        D3 --> D1 --> D2
    end

    subgraph REGISTRY["⑥ Model Registry"]
        E1["MLflow Registry — Staging → Production"]
        E2["Artifacts — Pickle / ONNX · scaler · encoder"]
        E1 --> E2
    end

    subgraph EVAL["⑦ Evaluation & Validation"]
        F1["Offline — Accuracy · F1 · Sharpe · MAE · Backtesting"]
        F2["Evidently AI — data drift · model performance drift"]
        F1 --> F2
    end

    subgraph SERVE["⑧ Model Serving"]
        G1["FastAPI — /predict/price-direction · /predict/ai-score"]
        G2["server.py on Railway — replaces hardcoded aiScore"]
        G3["LangGraph Agent — model predictions as MCP context"]
        G1 --> G2 --> G3
    end

    subgraph OUTPUT["⑩ EveryCoin Integration"]
        J1["Live AI Score — real model replaces hardcoded 84/100"]
        J2["Price Signal — BUY · HOLD · SELL from classifier"]
        J3["Agent Context — insights fed to strategist node"]
        J1 --> J2 --> J3
    end

    subgraph MONITOR["⑨ Monitoring"]
        H1["Evidently AI — feature & prediction drift reports"]
        H2["Prometheus + Grafana — latency · throughput · errors"]
        H3["Retraining Trigger — drift threshold crossed"]
        H1 --> H2 --> H3
    end

    subgraph CICD["CI/CD — GitHub Actions · Docker · Railway"]
        I1["Test → Train → Evaluate → Promote → Deploy"]
    end

    INGEST   --> RAWSTORE
    RAWSTORE --> FEATURES
    FEATURES --> FSTORE
    FSTORE   --> TRAIN
    TRAIN    --> REGISTRY
    REGISTRY --> EVAL
    EVAL     -->|"pass"| SERVE
    EVAL     -->|"fail → retrain"| TRAIN
    SERVE    --> OUTPUT
    SERVE    --> MONITOR
    MONITOR  -->|"trigger retrain"| TRAIN
    CICD     --> TRAIN
    CICD     --> SERVE
```

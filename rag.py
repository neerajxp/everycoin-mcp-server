"""
RAG Engine — ChromaDB + sentence-transformers
Loads knowledge/*.md at startup, chunks, embeds, stores persistently.
Query: semantic similarity search with optional topic filtering.
"""

import logging
import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger("everycoin.rag")

CHUNK_SIZE = 300      # tokens (approx — we split by words)
CHUNK_OVERLAP = 50
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "everycoin_knowledge"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Model loaded.")
    return _model


def _chunk_text(text: str, source: str, topic: str) -> list[dict]:
    """Split text into overlapping word-based chunks with metadata."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_text) > 50:  # skip tiny fragments
            chunks.append({
                "text": chunk_text,
                "id": f"{source}_{idx}",
                "metadata": {"source": source, "topic": topic},
            })
            idx += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _extract_topic(filename: str) -> str:
    mapping = {
        "defi_protocols": "defi",
        "security": "security",
        "market_strategy": "strategy",
        "l2_scaling": "l2",
    }
    stem = Path(filename).stem
    return mapping.get(stem, "general")


def init_rag() -> None:
    """Initialize ChromaDB and load knowledge base. Called once at startup."""
    global _client, _collection

    CHROMA_DIR.mkdir(exist_ok=True)
    _client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    existing = [c.name for c in _client.list_collections()]
    if COLLECTION_NAME in existing:
        _collection = _client.get_collection(COLLECTION_NAME)
        log.info("RAG: loaded existing collection (%d docs)", _collection.count())
        return

    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = _get_model()
    all_chunks: list[dict] = []

    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        topic = _extract_topic(md_file.name)
        text = md_file.read_text(encoding="utf-8")
        # strip markdown syntax for cleaner embeddings
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        chunks = _chunk_text(text, md_file.name, topic)
        all_chunks.extend(chunks)
        log.info("RAG: chunked %s → %d chunks (topic: %s)", md_file.name, len(chunks), topic)

    if not all_chunks:
        log.warning("RAG: no knowledge files found in %s", KNOWLEDGE_DIR)
        return

    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    log.info("RAG: embedding %d chunks...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    _collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    log.info("RAG: indexed %d chunks into ChromaDB", len(texts))


def search(query: str, topic: str | None = None, n_results: int = 3) -> list[dict]:
    """
    Semantic search over the knowledge base.
    Returns list of { text, source, topic, distance }.
    """
    if _collection is None:
        return [{"text": "RAG not initialized.", "source": "", "topic": ""}]

    model = _get_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    where = {"topic": topic} if topic else None

    try:
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, _collection.count() or 1),
            where=where,
        )
    except Exception:
        # fallback without filter if topic yields no results
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, _collection.count() or 1),
        )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "topic": meta.get("topic", ""),
            "distance": round(dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]

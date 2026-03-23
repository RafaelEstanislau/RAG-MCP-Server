from typing import Any, Callable

import chromadb
from chromadb import Collection
from sentence_transformers import SentenceTransformer

from config.settings import settings


_client: chromadb.PersistentClient | None = None
_collection: Collection | None = None
_model: SentenceTransformer | None = None


def _get_collection(
    client: chromadb.ClientAPI | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> Collection:
    """Return (or lazily create) the ChromaDB collection."""
    global _client, _collection, _model

    if client is not None:
        # Injected client (used in tests)
        return client.get_or_create_collection("references")

    if _collection is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
        _collection = _client.get_or_create_collection("references")

    return _collection


def _get_embedder(
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    global _model

    if embedder is not None:
        return embedder

    if _model is None:
        _model = SentenceTransformer(settings.embed_model)

    def _encode(texts: list[str]) -> list[list[float]]:
        return _model.encode(texts, convert_to_numpy=True).tolist()

    return _encode


def upsert_chunks(
    chunks: list[dict],
    client: chromadb.ClientAPI | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    """Embed and upsert chunks into the collection. Returns list of inserted chunk IDs."""
    if not chunks:
        return []

    collection = _get_collection(client)
    encode = _get_embedder(embedder)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "file_id": c["file_id"],
            "file_name": c["file_name"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    embeddings = encode(texts)

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return ids


def query_chunks(
    query: str,
    top_k: int = 5,
    client: chromadb.ClientAPI | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict]:
    """Semantic search. Returns list of {chunk_id, file_name, page_number, text, score} dicts."""
    collection = _get_collection(client)
    encode = _get_embedder(embedder)

    query_embedding = encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        output.append(
            {
                "chunk_id": doc_id,
                "file_name": meta["file_name"],
                "page_number": meta["page_number"],
                "text": results["documents"][0][i],
                "score": round(1 - distance, 4),  # cosine similarity approximation
            }
        )
    return output


def delete_by_file_id(
    file_id: str,
    client: chromadb.ClientAPI | None = None,
) -> None:
    """Delete all chunks belonging to a given file_id."""
    collection = _get_collection(client)
    results = collection.get(where={"file_id": file_id}, include=[])
    if results["ids"]:
        collection.delete(ids=results["ids"])


def list_papers(
    client: chromadb.ClientAPI | None = None,
) -> list[dict]:
    """Return unique paper metadata (file_id, file_name, chunk count)."""
    collection = _get_collection(client)
    all_items = collection.get(include=["metadatas"])

    seen: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        fid = meta["file_id"]
        if fid not in seen:
            seen[fid] = {"file_id": fid, "file_name": meta["file_name"], "chunk_count": 0}
        seen[fid]["chunk_count"] += 1

    return list(seen.values())

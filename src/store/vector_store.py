import uuid
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config.settings import settings


COLLECTION_NAME = "references"
VECTOR_SIZE = 384

_client: QdrantClient | None = None
_model: SentenceTransformer | None = None


def _chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _get_client(client: QdrantClient | None = None) -> QdrantClient:
    global _client
    if client is not None:
        return client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="file_id",
        field_schema="keyword",
    )


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
    client: QdrantClient | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    """Embed and upsert chunks into the collection. Returns list of inserted chunk IDs."""
    if not chunks:
        return []

    qdrant = _get_client(client)
    encode = _get_embedder(embedder)

    texts = [c["text"] for c in chunks]
    embeddings = encode(texts)

    points = [
        PointStruct(
            id=_chunk_id_to_uuid(c["chunk_id"]),
            vector=embeddings[i],
            payload={
                "chunk_id": c["chunk_id"],
                "file_id": c["file_id"],
                "file_name": c["file_name"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "modified_time": c.get("modified_time", ""),
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return [c["chunk_id"] for c in chunks]


def query_chunks(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict]:
    """Semantic search. Returns list of {chunk_id, file_name, page_number, text, score} dicts."""
    qdrant = _get_client(client)
    encode = _get_embedder(embedder)

    query_embedding = encode([query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    return [
        {
            "chunk_id": r.payload["chunk_id"],
            "file_name": r.payload["file_name"],
            "page_number": r.payload["page_number"],
            "text": r.payload["text"],
            "score": round(r.score, 4),
        }
        for r in response.points
    ]


def delete_by_file_id(
    file_id: str,
    client: QdrantClient | None = None,
) -> None:
    """Delete all chunks belonging to a given file_id."""
    qdrant = _get_client(client)
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            )
        ),
    )


def list_papers(
    client: QdrantClient | None = None,
) -> list[dict]:
    """Return unique paper metadata (file_id, file_name, chunk count)."""
    qdrant = _get_client(client)
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[str, dict] = {}
    for point in points:
        fid = point.payload["file_id"]
        if fid not in seen:
            seen[fid] = {
                "file_id": fid,
                "file_name": point.payload["file_name"],
                "chunk_count": 0,
            }
        seen[fid]["chunk_count"] += 1

    return list(seen.values())


def list_indexed_files(
    client: QdrantClient | None = None,
) -> dict[str, str]:
    """Return {file_id: modified_time} for all indexed files."""
    qdrant = _get_client(client)
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    result: dict[str, str] = {}
    for point in points:
        fid = point.payload["file_id"]
        if fid not in result:
            result[fid] = point.payload.get("modified_time", "")

    return result

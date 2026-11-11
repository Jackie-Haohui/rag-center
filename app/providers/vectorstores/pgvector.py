from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.core.index_versions import DEFAULT_INDEX_VERSION
from app.models.chunk import Chunk
from app.providers.vectorstores.base import VectorStore
from app.repositories.chunk_repository import ChunkRepository


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, db: Session) -> None:
        self.db = db
        self.chunk_repository = ChunkRepository(db)

    async def add_chunks(self, chunks: list[dict]) -> None:
        db_chunks = [
            Chunk(
                id=chunk["chunk_id"],
                tenant_id=chunk["tenant_id"],
                kb_id=chunk["kb_id"],
                document_id=chunk["document_id"],
                title=chunk["title"],
                content=chunk["content"],
                retrieval_text=chunk.get("retrieval_text"),
                index_version=chunk.get("index_version", DEFAULT_INDEX_VERSION),
                section_id=chunk.get("section_id"),
                parent_section_id=chunk.get("parent_section_id"),
                order_index=chunk.get("order_index"),
                metadata_=chunk.get("metadata", {}),
                embedding=chunk["embedding"],
            )
            for chunk in chunks
        ]
        self.chunk_repository.add_many(db_chunks)

    async def similarity_search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        kb_id: str,
        index_version: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        effective_version = index_version or DEFAULT_INDEX_VERSION
        return await self.similarity_search_scope(
            query_vector,
            tenant_id=tenant_id,
            index_versions={kb_id: effective_version},
            top_k=top_k,
        )

    async def similarity_search_scope(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        index_versions: dict[str, str],
        top_k: int = 5,
    ) -> list[dict]:
        if not index_versions:
            return []
        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
        version_scope = or_(
            *[
                and_(Chunk.kb_id == kb_id, Chunk.index_version == index_version)
                for kb_id, index_version in sorted(index_versions.items())
            ]
        )
        stmt = select(Chunk, distance).where(
            Chunk.tenant_id == tenant_id,
            version_scope,
        )
        stmt = stmt.where(
            or_(
                Chunk.metadata_["chunk_type"].astext.is_(None),
                Chunk.metadata_["chunk_type"].astext != "reference_pointer",
            )
        )
        stmt = stmt.order_by(distance, Chunk.kb_id, Chunk.id).limit(top_k)
        rows = self.db.execute(stmt).all()

        results: list[dict] = []
        for chunk, distance_value in rows:
            score = 1.0 - float(distance_value)
            results.append(
                {
                    "kb_id": chunk.kb_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "retrieval_text": chunk.retrieval_text,
                    "index_version": chunk.index_version,
                    "section_id": chunk.section_id,
                    "parent_section_id": chunk.parent_section_id,
                    "order_index": chunk.order_index,
                    "score": score,
                    "metadata": chunk.metadata_ or {},
                }
            )
        return results

    async def delete_by_document_id(
        self, document_id: str, *, index_version: str | None = None
    ) -> None:
        self.chunk_repository.delete_by_document_id(document_id, index_version=index_version)

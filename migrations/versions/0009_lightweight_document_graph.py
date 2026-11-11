"""lightweight document graph and index versions

Revision ID: 0009_lightweight_document_graph
Revises: 0008_document_source_file
Create Date: 2026-08-15
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0009_lightweight_document_graph"
down_revision: str | None = "0008_document_source_file"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column("active_index_version", sa.String(length=32), server_default="v1", nullable=False),
    )

    op.create_table(
        "knowledge_base_index_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("document_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'active', 'failed', 'archived')",
            name="ck_kb_index_versions_status",
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id", "kb_id", "version", name="uq_kb_index_versions_tenant_kb_version"
        ),
    )
    op.create_index(
        "ix_kb_index_versions_tenant_kb",
        "knowledge_base_index_versions",
        ["tenant_id", "kb_id"],
    )
    op.create_index(
        "uq_kb_index_versions_active",
        "knowledge_base_index_versions",
        ["tenant_id", "kb_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.add_column(
        "chunks",
        sa.Column("index_version", sa.String(length=32), server_default="v1", nullable=True),
    )
    op.add_column("chunks", sa.Column("retrieval_text", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("section_id", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("parent_section_id", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("order_index", sa.Integer(), nullable=True))

    op.execute("UPDATE chunks SET index_version = 'v1' WHERE index_version IS NULL")
    op.alter_column("chunks", "index_version", nullable=False, server_default="v1")
    op.create_index(
        "ix_chunks_tenant_kb_version",
        "chunks",
        ["tenant_id", "kb_id", "index_version"],
    )
    op.create_index(
        "ix_chunks_document_version_order",
        "chunks",
        ["document_id", "index_version", "order_index"],
    )
    op.create_index(
        "ix_chunks_document_version_section",
        "chunks",
        ["document_id", "index_version", "section_id"],
    )
    op.create_unique_constraint(
        "uq_chunks_document_version_order",
        "chunks",
        ["document_id", "index_version", "order_index"],
    )

    op.execute(
        """
        INSERT INTO knowledge_base_index_versions (
            id, tenant_id, kb_id, version, status, document_count, chunk_count,
            error_message, created_at, completed_at, activated_at
        )
        SELECT
            md5(kb.id || chr(58) || 'v1'), kb.tenant_id, kb.id, 'v1', 'active',
            (SELECT count(*) FROM documents d WHERE d.kb_id = kb.id),
            (SELECT count(*) FROM chunks c WHERE c.kb_id = kb.id AND c.index_version = 'v1'),
            NULL, COALESCE(kb.created_at, now()), now(), now()
        FROM knowledge_bases kb
        ON CONFLICT (tenant_id, kb_id, version) DO NOTHING
        """
    )

    op.create_table(
        "chunk_relations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("index_version", sa.String(length=32), nullable=False),
        sa.Column("source_chunk_id", sa.String(length=36), nullable=False),
        sa.Column("target_chunk_id", sa.String(length=36), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("source_chunk_id <> target_chunk_id", name="ck_chunk_relations_not_self"),
        sa.CheckConstraint("relation_type = 'reference'", name="ck_chunk_relations_type"),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_chunk_id",
            "target_chunk_id",
            "relation_type",
            name="uq_chunk_relations_source_target_type",
        ),
    )
    op.create_index(
        "ix_chunk_relations_tenant_kb_version",
        "chunk_relations",
        ["tenant_id", "kb_id", "index_version"],
    )
    op.create_index("ix_chunk_relations_source", "chunk_relations", ["source_chunk_id"])
    op.create_index("ix_chunk_relations_target", "chunk_relations", ["target_chunk_id"])


def downgrade() -> None:
    op.drop_table("chunk_relations")
    op.drop_constraint("uq_chunks_document_version_order", "chunks", type_="unique")
    op.drop_index("ix_chunks_document_version_section", table_name="chunks")
    op.drop_index("ix_chunks_document_version_order", table_name="chunks")
    op.drop_index("ix_chunks_tenant_kb_version", table_name="chunks")
    op.drop_column("chunks", "order_index")
    op.drop_column("chunks", "parent_section_id")
    op.drop_column("chunks", "section_id")
    op.drop_column("chunks", "retrieval_text")
    op.drop_column("chunks", "index_version")
    op.drop_index("uq_kb_index_versions_active", table_name="knowledge_base_index_versions")
    op.drop_index("ix_kb_index_versions_tenant_kb", table_name="knowledge_base_index_versions")
    op.drop_table("knowledge_base_index_versions")
    op.drop_column("knowledge_bases", "active_index_version")

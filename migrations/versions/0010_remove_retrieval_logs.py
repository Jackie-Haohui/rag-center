"""remove redundant local retrieval logs

Revision ID: 0010_remove_retrieval_logs
Revises: 0009_lightweight_document_graph
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_remove_retrieval_logs"
down_revision: str | None = "0009_lightweight_document_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_table("retrieval_logs")


def downgrade() -> None:
    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column("kb_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("profile", sa.String(length=32), nullable=True),
        sa.Column("search_query", sa.Text(), nullable=True),
        sa.Column("effective_query", sa.Text(), nullable=True),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column(
            "retrieved_chunks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("vector_store", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retrieval_logs_tenant_id", "retrieval_logs", ["tenant_id"])
    op.create_index("ix_retrieval_logs_kb_id", "retrieval_logs", ["kb_id"])
    op.create_index(
        "ix_retrieval_logs_tenant_kb",
        "retrieval_logs",
        ["tenant_id", "kb_id"],
    )

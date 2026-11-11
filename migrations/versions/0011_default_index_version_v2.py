"""Default new knowledge bases and chunks to index version v2.

Revision ID: 0011_default_index_version_v2
Revises: 0010_remove_retrieval_logs
"""

from alembic import op

revision: str = "0011_default_index_version_v2"
down_revision: str | None = "0010_remove_retrieval_logs"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.alter_column("knowledge_bases", "active_index_version", server_default="v2")
    op.alter_column("chunks", "index_version", server_default="v2")


def downgrade() -> None:
    op.alter_column("chunks", "index_version", server_default="v1")
    op.alter_column("knowledge_bases", "active_index_version", server_default="v1")

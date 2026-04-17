"""Add user_id to token_usage for per-user aggregation.

Revision ID: 0002_token_usage_user_id
Revises: 0001_initial
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_token_usage_user_id"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "token_usage",
        sa.Column("user_id", sa.String(length=64), nullable=False, server_default=""),
    )
    op.create_index(
        "ix_usage_user_model_created",
        "token_usage",
        ["user_id", "model", "created_at"],
        unique=False,
    )
    op.alter_column("token_usage", "user_id", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_usage_user_model_created", table_name="token_usage")
    op.drop_column("token_usage", "user_id")

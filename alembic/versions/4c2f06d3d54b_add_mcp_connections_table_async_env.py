"""add_mcp_connections_table_async_env

Revision ID: 4c2f06d3d54b
Revises: 73cac0cbb643
Create Date: 2025-05-09 16:04:39.510759

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '4c2f06d3d54b'
down_revision: Union[str, None] = '73cac0cbb643'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table('mcp_connections',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('mcp_server_url', sa.String(), nullable=False),
    sa.Column('alias', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_mcp_connections_alias'), 'mcp_connections', ['alias'], unique=False)
    op.create_index(op.f('ix_mcp_connections_id'), 'mcp_connections', ['id'], unique=False)
    op.create_index(op.f('ix_mcp_connections_mcp_server_url'), 'mcp_connections', ['mcp_server_url'], unique=False)
    op.create_index(op.f('ix_mcp_connections_user_id'), 'mcp_connections', ['user_id'], unique=False)



def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f('ix_mcp_connections_user_id'), table_name='mcp_connections')
    op.drop_index(op.f('ix_mcp_connections_mcp_server_url'), table_name='mcp_connections')
    op.drop_index(op.f('ix_mcp_connections_id'), table_name='mcp_connections')
    op.drop_index(op.f('ix_mcp_connections_alias'), table_name='mcp_connections')
    op.drop_table('mcp_connections')


"""add_memory_entries_table

Revision ID: 73cac0cbb643
Revises: d1281cd6ea7c
Create Date: 2025-05-08 22:58:02.332780

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '73cac0cbb643'
down_revision: Union[str, None] = 'd1281cd6ea7c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table('memory_entries',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=True),
    sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('memory_type', sa.String(), nullable=False),
    sa.Column('scope', sa.String(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('source_message_ids', sa.JSON(), nullable=True),
    sa.Column('keywords', sa.JSON(), nullable=True),
    sa.Column('importance', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_memory_entries_id'), 'memory_entries', ['id'], unique=False)
    op.create_index(op.f('ix_memory_entries_memory_type'), 'memory_entries', ['memory_type'], unique=False)
    op.create_index(op.f('ix_memory_entries_scope'), 'memory_entries', ['scope'], unique=False)
    op.create_index(op.f('ix_memory_entries_session_id'), 'memory_entries', ['session_id'], unique=False)
    op.create_index(op.f('ix_memory_entries_timestamp'), 'memory_entries', ['timestamp'], unique=False)
    op.create_index(op.f('ix_memory_entries_user_id'), 'memory_entries', ['user_id'], unique=False)



def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f('ix_memory_entries_user_id'), table_name='memory_entries')
    op.drop_index(op.f('ix_memory_entries_timestamp'), table_name='memory_entries')
    op.drop_index(op.f('ix_memory_entries_session_id'), table_name='memory_entries')
    op.drop_index(op.f('ix_memory_entries_scope'), table_name='memory_entries')
    op.drop_index(op.f('ix_memory_entries_memory_type'), table_name='memory_entries')
    op.drop_index(op.f('ix_memory_entries_id'), table_name='memory_entries')
    op.drop_table('memory_entries')


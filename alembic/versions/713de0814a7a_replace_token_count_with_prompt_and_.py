"""replace_token_count_with_prompt_and_completion_tokens_in_messages

Revision ID: 713de0814a7a
Revises: 
Create Date: 2025-05-08 02:13:58.227880

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '713de0814a7a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table('users',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_table('api_keys',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('model_provider', sa.String(), nullable=False),
    sa.Column('encrypted_api_key', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_id'), 'api_keys', ['id'], unique=False)
    op.create_index(op.f('ix_api_keys_model_provider'), 'api_keys', ['model_provider'], unique=False)
    op.create_table('sessions',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('alias', sa.String(), nullable=True),
    sa.Column('system_prompt', sa.Text(), nullable=True),
    sa.Column('memory_mode', sa.Enum('AUTO', 'EXPLICIT', 'DISABLED', name='memorymode_enum', create_constraint=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sessions_alias'), 'sessions', ['alias'], unique=False)
    op.create_index(op.f('ix_sessions_id'), 'sessions', ['id'], unique=False)
    op.create_table('messages',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('parent_message_id', sa.Integer(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('role', sa.Enum('USER', 'ASSISTANT', 'SYSTEM', name='messagerole_enum', create_constraint=True), nullable=False),
    sa.Column('content_text', sa.Text(), nullable=False),
    sa.Column('model_used', sa.String(), nullable=True),
    sa.Column('token_count', sa.Integer(), nullable=True),
    sa.Column('response_time_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.ForeignKeyConstraint(['parent_message_id'], ['messages.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)



def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f('ix_messages_id'), table_name='messages')
    op.drop_table('messages')
    op.drop_index(op.f('ix_sessions_id'), table_name='sessions')
    op.drop_index(op.f('ix_sessions_alias'), table_name='sessions')
    op.drop_table('sessions')
    op.drop_index(op.f('ix_api_keys_model_provider'), table_name='api_keys')
    op.drop_index(op.f('ix_api_keys_id'), table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')


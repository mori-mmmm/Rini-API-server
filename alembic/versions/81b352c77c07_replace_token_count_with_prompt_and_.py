"""replace_token_count_with_prompt_and_completion_tokens_in_messages

Revision ID: 81b352c77c07
Revises: 713de0814a7a
Create Date: 2025-05-08 02:23:15.134018

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '81b352c77c07'
down_revision: Union[str, None] = '713de0814a7a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column('messages', sa.Column('prompt_tokens', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('completion_tokens', sa.Integer(), nullable=True))



def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column('messages', 'completion_tokens')
    op.drop_column('messages', 'prompt_tokens')


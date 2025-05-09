"""add_file_reference_and_metadata_to_messages

Revision ID: d1281cd6ea7c
Revises: 82f8bdd80dc5
Create Date: 2025-05-08 17:06:30.429408

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = 'd1281cd6ea7c'
down_revision: Union[str, None] = '82f8bdd80dc5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('messages', sa.Column('file_reference', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('original_filename', sa.String(), nullable=True))
    op.add_column('messages', sa.Column('mime_type', sa.String(), nullable=True))
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.alter_column('content',
                          existing_type=sa.Text(),
                          nullable=True)
    op.create_index(op.f('ix_messages_file_reference'), 'messages', ['file_reference'], unique=False)



def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f('ix_messages_file_reference'), table_name='messages')
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.alter_column('content',
                          existing_type=sa.Text(),
                          nullable=False)
    op.drop_column('messages', 'mime_type')
    op.drop_column('messages', 'original_filename')
    op.drop_column('messages', 'file_reference')
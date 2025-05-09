"""rename_message_content_text_to_content

Revision ID: 82f8bdd80dc5
Revises: 81b352c77c07
Create Date: 2025-05-08 08:40:46.383965

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = '82f8bdd80dc5'
down_revision: Union[str, None] = '81b352c77c07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:


    with op.batch_alter_table('messages', schema=None) as batch_op:

        batch_op.add_column(sa.Column('content', sa.Text(), nullable=True))



    op.execute('UPDATE messages SET content = content_text')


    with op.batch_alter_table('messages', schema=None) as batch_op:


        batch_op.alter_column('content',
                              existing_type=sa.Text(),
                              nullable=False)


    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('content_text')



def downgrade() -> None:

    with op.batch_alter_table('messages', schema=None) as batch_op:

        batch_op.add_column(sa.Column('content_text', sa.Text(), nullable=True))


    op.execute('UPDATE messages SET content_text = content')


    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.alter_column('content_text',
                              existing_type=sa.Text(),
                              nullable=False)


    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('content')


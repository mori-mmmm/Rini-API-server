import asyncio
from logging.config import fileConfig

import os
import sys
from sqlalchemy import engine_from_config
from sqlalchemy import pool
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.models import Base

from app.database import engine as async_engine_from_app

from alembic import context



config = context.config





if config.config_file_name is not None:
    fileConfig(config.config_file_name)





target_metadata = Base.metadata







def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_app


    async with connectable.connect() as connection:

        await connection.run_sync(do_run_migrations)


    await connectable.dispose()

def do_run_migrations(connection):
    """Alembic의 context를 설정하고 마이그레이션을 실행하는 동기 함수"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata

    )

    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

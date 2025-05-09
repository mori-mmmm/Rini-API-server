

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger("app.database")

SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./rini_api.db"
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
logger.info(f"Async database engine created for URL: {SQLALCHEMY_DATABASE_URL}")


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)
logger.info("Async local session factory configured.")

Base = declarative_base()
logger.info("Declarative base created.")

async def get_db() -> AsyncSession:
    logger.debug("Creating new async database session.")
    async_db = AsyncSessionLocal()
    try:
        yield async_db
    finally:
        logger.debug("Closing async database session.")
        await async_db.close()

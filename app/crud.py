

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func as sql_func, Date
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, date, time
import logging
from enum import Enum


from . import models, schemas, security
from .config import MODEL_PRICING
from .schemas import MemoryScope, MemoryType


logger = logging.getLogger("app.crud")





async def get_user(db: AsyncSession, user_id: int) -> Optional[models.User]:
    """주어진 ID를 가진 사용자를 데이터베이스에서 비동기적으로 조회합니다."""
    logger.debug(f"Querying user with ID: {user_id}")
    result = await db.execute(select(models.User).filter(models.User.id == user_id))
    user = result.scalars().first()
    if user:
        logger.debug(f"User found: {user.id}")
    else:
        logger.debug(f"User not found with ID: {user_id}")
    return user

async def create_user(db: AsyncSession) -> models.User:
    """새로운 사용자를 비동기적으로 생성합니다."""
    logger.debug("Creating a new user instance.")
    db_user = models.User()
    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        logger.info(f"User created successfully with ID: {db_user.id}")
        return db_user
    except Exception as e:
        await db.rollback()
        logger.exception(f"Failed to create user: {e}")
        raise




async def get_api_key(db: AsyncSession, api_key_id: int, user_id: int) -> Optional[models.ApiKey]:
    """주어진 ID와 사용자 ID를 가진 API 키를 비동기적으로 조회합니다."""
    logger.debug(f"Querying API key with ID: {api_key_id} for user ID: {user_id}")
    stmt = select(models.ApiKey).filter(models.ApiKey.id == api_key_id, models.ApiKey.user_id == user_id)
    result = await db.execute(stmt)
    api_key = result.scalars().first()
    if api_key:
        logger.debug(f"API key found: {api_key.id}, Provider: {api_key.model_provider}")
    else:
        logger.debug(f"API key not found or not authorized for ID: {api_key_id}, User ID: {user_id}")
    return api_key

async def get_api_keys_by_user(db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[models.ApiKey]:
    """특정 사용자의 모든 API 키 목록을 비동기적으로 조회합니다."""
    logger.debug(f"Querying API keys for user ID: {user_id} with skip: {skip}, limit: {limit}")
    stmt = select(models.ApiKey)\
             .filter(models.ApiKey.user_id == user_id)\
             .offset(skip)\
             .limit(limit)
    result = await db.execute(stmt)
    api_keys = result.scalars().all()
    logger.debug(f"Found {len(api_keys)} API keys for user ID: {user_id}")
    return api_keys

async def create_api_key(db: AsyncSession, api_key: schemas.ApiKeyCreate, user_id: int) -> models.ApiKey:
    """새로운 API 키를 생성하고 암호화하여 비동기적으로 저장합니다."""
    logger.debug(f"Attempting to create API key for user ID: {user_id}, provider: {api_key.model_provider}")
    try:
        logger.debug("Encrypting API key value.")
        encrypted_key = security.encrypt_api_key(api_key.api_key_value)
        logger.debug("API key value encrypted.")

        db_api_key = models.ApiKey(
            model_provider=api_key.model_provider,
            description=api_key.description,
            _encrypted_api_key=encrypted_key,
            user_id=user_id
        )
        db.add(db_api_key)
        await db.commit()
        await db.refresh(db_api_key)
        logger.info(f"API key created successfully with ID: {db_api_key.id} for user ID: {user_id}")
        return db_api_key
    except Exception as e:
        await db.rollback()
        logger.exception(f"Failed to create API key for user ID {user_id}, provider {api_key.model_provider}: {e}")
        raise

async def update_api_key(db: AsyncSession, api_key_id: int, api_key_update: schemas.ApiKeyUpdate, user_id: int) -> Optional[models.ApiKey]:
    """기존 API 키의 정보를 비동기적으로 수정합니다."""
    logger.debug(f"Attempting to update API key ID: {api_key_id} for user ID: {user_id}")
    db_api_key = await get_api_key(db, api_key_id=api_key_id, user_id=user_id)
    if db_api_key:
        update_data = api_key_update.model_dump(exclude_unset=True)
        logger.debug(f"Update data for API key {api_key_id}: {update_data}")
        if not update_data:
             logger.warning(f"No update data provided for API key {api_key_id}.")
             return db_api_key

        try:
            for key, value in update_data.items():
                setattr(db_api_key, key, value)
            await db.commit()
            await db.refresh(db_api_key)
            logger.info(f"API key {api_key_id} updated successfully for user ID: {user_id}")
            return db_api_key
        except Exception as e:
            await db.rollback()
            logger.exception(f"Failed to update API key {api_key_id} for user ID {user_id}: {e}")
            raise
    else:
        logger.warning(f"API key {api_key_id} not found or not authorized for update by user {user_id}.")
        return None

async def delete_api_key(db: AsyncSession, api_key_id: int, user_id: int) -> Optional[models.ApiKey]:
    """API 키를 비동기적으로 삭제합니다."""
    logger.debug(f"Attempting to delete API key ID: {api_key_id} for user ID: {user_id}")
    db_api_key = await get_api_key(db, api_key_id=api_key_id, user_id=user_id)
    if db_api_key:
        try:
            await db.delete(db_api_key)
            await db.commit()
            logger.info(f"API key {api_key_id} deleted successfully for user ID: {user_id}")
            return db_api_key
        except Exception as e:
            await db.rollback()
            logger.exception(f"Failed to delete API key {api_key_id} for user ID {user_id}: {e}")
            raise
    else:
        logger.warning(f"API key {api_key_id} not found or not authorized for deletion by user {user_id}.")
        return None

async def get_session(db: AsyncSession, session_id: str, user_id: int) -> Optional[models.Session]:
    """주어진 ID와 사용자 ID를 가진 세션을 비동기적으로 조회합니다."""
    stmt = select(models.Session).filter(models.Session.id == session_id, models.Session.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_sessions_by_user(db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Session]:
    """특정 사용자의 모든 세션 목록을 비동기적으로 조회합니다."""
    stmt = select(models.Session)\
             .filter(models.Session.user_id == user_id)\
             .order_by(models.Session.created_at.desc())\
             .offset(skip)\
             .limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_session(db: AsyncSession, session: schemas.SessionCreate, user_id: int) -> models.Session:
    """새로운 세션을 비동기적으로 생성합니다."""
    session_id = str(uuid.uuid4())
    db_session = models.Session(
        id=session_id,
        **session.model_dump(),
        user_id=user_id
    )
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session

async def update_session(db: AsyncSession, session_id: str, session_update: schemas.SessionUpdate, user_id: int) -> Optional[models.Session]:
    """세션 정보를 비동기적으로 수정합니다."""
    db_session = await get_session(db, session_id=session_id, user_id=user_id)
    if db_session:
        update_data = session_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_session, key, value)
        await db.commit()
        await db.refresh(db_session)
    return db_session

async def delete_session(db: AsyncSession, session_id: str, user_id: int) -> Optional[models.Session]:
    """세션을 비동기적으로 삭제합니다."""
    db_session = await get_session(db, session_id=session_id, user_id=user_id)
    if db_session:
        await db.delete(db_session)
        await db.commit()
    return db_session




async def get_message(db: AsyncSession, message_id: int, user_id: int) -> Optional[models.Message]:
    """
    주어진 message_id부터 시작하여 부모 메시지를 거슬러 올라가 루트까지의
    모든 메시지 목록을 시간 순서대로 반환합니다.
    요청한 사용자가 해당 메시지 체인이 속한 세션의 소유주일 경우에만 반환합니다.
    """
    logger.debug(f"Querying message with ID: {message_id}")
    stmt = select(models.Message).filter(models.Message.id == message_id)
    result = await db.execute(stmt)
    message = result.scalars().first()
    if not message:
        logger.debug(f"Message not found with ID: {message_id}")
        return None
    
    logger.debug(f"Checking session ownership for message {message_id} (session: {message.session_id}) against user ID: {user_id}")
    owner_stmt = select(models.Session.user_id).filter(models.Session.id == message.session_id)
    owner_result = await db.execute(owner_stmt)
    session_owner_id = owner_result.scalar_one_or_none()
    if session_owner_id == user_id:
        logger.debug(f"Message {message_id} access granted for user {user_id}.")
        return message
    else:
        logger.warning(f"Message {message_id} access denied for user {user_id}. Session owner is {session_owner_id}.")
        return None

async def get_messages_by_session(
        db: AsyncSession,
        session_id: str,
        user_id: int,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[models.Message]:
    """특정 세션의 메시지 목록을 비동기적으로 조회합니다."""
    logger.debug(f"Querying messages for session ID: {session_id}, user ID: {user_id}, skip: {skip}, limit: {limit}")
    db_session = await get_session(db, session_id=session_id, user_id=user_id)
    if not db_session:
        logger.warning(f"Session {session_id} not found or not authorized for user {user_id} when querying messages.")
        return []

    stmt = select(models.Message)\
              .filter(models.Message.session_id == session_id)\
              .order_by(models.Message.created_at.asc())

    if skip > 0:
        stmt = stmt.offset(skip)
        logger.debug(f"Applying offset: {skip}")
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
        logger.debug(f"Applying limit: {limit}")

    result = await db.execute(stmt)
    messages = result.scalars().all()
    logger.debug(f"Found {len(messages)} messages for session {session_id}")
    return messages

async def create_message(db: AsyncSession, message_data: schemas.MessageCreateInternal, requesting_user_id: int) -> models.Message:
    """새로운 메시지를 데이터베이스에 비동기적으로 저장합니다."""
    logger.debug(f"Attempting to create message in session {message_data.session_id} by user {requesting_user_id}. Role: {message_data.role}, Parent: {message_data.parent_message_id}")
    db_session = await get_session(db, session_id=message_data.session_id, user_id=requesting_user_id)
    if not db_session:
        logger.warning(f"Attempt to create message in session {message_data.session_id} failed: Session not found or not authorized for user {requesting_user_id}.")



    logger.debug(f"Creating message model instance for session {message_data.session_id}")
    db_message = models.Message(
        session_id=message_data.session_id,
        parent_message_id=message_data.parent_message_id,
        user_id=message_data.user_id,
        role=message_data.role,
        content=message_data.content
    )
    try:
        db.add(db_message)
        await db.commit()
        await db.refresh(db_message)
        logger.info(f"Message created successfully with ID: {db_message.id} in session {message_data.session_id}")
        return db_message
    except Exception as e:
        await db.rollback()
        logger.exception(f"Failed to create message in session {message_data.session_id}: {e}")
        raise

async def get_last_message_in_session(db: AsyncSession, session_id: str) -> Optional[models.Message]:
    """특정 세션의 가장 마지막 메시지를 비동기적으로 조회합니다."""
    logger.debug(f"Querying last message for session ID: {session_id}")
    stmt = select(models.Message)\
                .filter(models.Message.session_id == session_id)\
                .order_by(models.Message.created_at.desc())\
                .limit(1)
    result = await db.execute(stmt)
    message = result.scalars().first()
    if message:
        logger.debug(f"Last message found for session {session_id}: ID {message.id}")
    else:
        logger.debug(f"No messages found for session {session_id}.")
    return message

async def get_message_parent_history(db: AsyncSession, message_id: int, user_id: int) -> List[models.Message]:
    """주어진 message_id부터 부모 메시지를 거슬러 올라가 루트까지의 모든 메시지 목록을 비동기적으로 반환합니다."""
    logger.debug(f"Getting message parent history starting from message ID: {message_id} for user ID: {user_id}")
    history = []
    current_message = await get_message(db, message_id=message_id, user_id=user_id)

    if not current_message:
        logger.warning(f"Initial message {message_id} not found or not authorized for user {user_id} when getting history.")
        return []

    logger.debug(f"Starting history traversal from message {current_message.id}")
    while current_message:
        logger.debug(f"Adding message {current_message.id} to history.")
        history.append(current_message)
        if current_message.parent_message_id:
            logger.debug(f"Looking for parent message ID: {current_message.parent_message_id}")
            parent_stmt = select(models.Message).filter(models.Message.id == current_message.parent_message_id)
            parent_result = await db.execute(parent_stmt)
            parent = parent_result.scalars().first()
            if parent and parent.session_id == current_message.session_id:
                logger.debug(f"Found parent message {parent.id}.")
                current_message = parent
            else:
                logger.error(f"Parent message (ID: {current_message.parent_message_id}) not found or session mismatch for message {current_message.id}. Stopping history traversal.")
                break
        else:
            logger.debug(f"Message {current_message.id} is the root message. Ending traversal.")
            break

    reversed_history = history[::-1]
    logger.info(f"Retrieved {len(reversed_history)} messages in history for starting message {message_id}.")
    return reversed_history













async def get_api_key_by_provider(db: AsyncSession, user_id: int, provider: str) -> Optional[models.ApiKey]:
    """특정 사용자의 지정된 제공사 API 키를 비동기적으로 조회합니다."""
    logger.debug(f"Querying API key for user ID: {user_id}, provider: {provider}")
    stmt = select(models.ApiKey)\
                .filter(models.ApiKey.user_id == user_id, models.ApiKey.model_provider == provider)
    result = await db.execute(stmt)
    api_key = result.scalars().first()
    if api_key:
        logger.debug(f"API key found for provider {provider}, user {user_id}. Key ID: {api_key.id}")
    else:
        logger.debug(f"API key not found for provider {provider}, user {user_id}.")
    return api_key


async def calculate_estimated_cost(
    db: AsyncSession,
    user_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """지정된 기간 및 필터 조건에 따라 예상 LLM 사용 비용을 비동기적으로 계산합니다."""

    stmt = select(
                models.Message.model_used,
                sql_func.sum(models.Message.prompt_tokens).label("total_prompt_tokens"),
                sql_func.sum(models.Message.completion_tokens).label("total_completion_tokens")
            )\
            .filter(models.Message.user_id == user_id, models.Message.role == schemas.MessageRole.ASSISTANT)

    if start_date:
        start_datetime = datetime.combine(start_date, time.min)
        stmt = stmt.filter(models.Message.created_at >= start_datetime)
    if end_date:
        end_datetime = datetime.combine(end_date, time.max)
        stmt = stmt.filter(models.Message.created_at <= end_datetime)
    if session_id:
        stmt = stmt.filter(models.Message.session_id == session_id)
    if model:
        stmt = stmt.filter(models.Message.model_used == model)
    elif provider:
        if provider == schemas.LLMProvider.GOOGLE.value:
            stmt = stmt.filter(models.Message.model_used.like('gemini%'))
        elif provider == schemas.LLMProvider.OPENAI.value:
            stmt = stmt.filter(models.Message.model_used.like('gpt%'))
        elif provider == schemas.LLMProvider.ANTHROPIC.value:
            stmt = stmt.filter(models.Message.model_used.like('claude%'))

    stmt = stmt.group_by(models.Message.model_used)
    result = await db.execute(stmt)
    usage_by_model = result.all()

    total_cost = 0.0
    details = {}
    for row in usage_by_model:
        model_used, total_prompt, total_comp = row.model_used, row.total_prompt_tokens, row.total_completion_tokens
        if not model_used or (total_prompt is None and total_comp is None):
            continue

        prompt_tokens = int(total_prompt or 0)
        completion_tokens = int(total_comp or 0)

        pricing = MODEL_PRICING.get(model_used, {})
        prompt_price = pricing.get("prompt", 0.0)
        completion_price = pricing.get("completion", 0.0)

        model_cost = ((prompt_tokens / 1_000_000) * prompt_price) + \
                     ((completion_tokens / 1_000_000) * completion_price)

        total_cost += model_cost
        details[model_used] = schemas.CostEstimationDetail(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=round(model_cost, 6)
        )

    return {
        "total_estimated_cost": round(total_cost, 6),
        "currency": "USD",
        "start_date": start_date,
        "end_date": end_date,
        "details": details
    }


async def create_memory_entry(
    db: AsyncSession,
    memory_data: schemas.MemoryEntryCreate,
    user_id: int,
    session_id: Optional[str] = None
) -> models.MemoryEntry:
    """새로운 메모리 항목을 비동기적으로 생성합니다."""
    effective_session_id = session_id if memory_data.scope == MemoryScope.SESSION else None

    db_memory = models.MemoryEntry(
        user_id=user_id,
        session_id=effective_session_id,
        memory_type=memory_data.memory_type.value,
        scope=memory_data.scope.value,
        content=memory_data.content,
        source_message_ids=memory_data.source_message_ids,
        keywords=memory_data.keywords,
        importance=memory_data.importance
    )
    db.add(db_memory)
    await db.commit()
    await db.refresh(db_memory)
    logger.info(f"Memory entry created (ID: {db_memory.id}, Scope: {db_memory.scope}, Type: {db_memory.memory_type}) for user {user_id}")
    return db_memory

async def get_memory_entries(
    db: AsyncSession,
    user_id: int,
    session_id: Optional[str] = None,
    scope: Optional[schemas.MemoryScope] = None,
    memory_type: Optional[schemas.MemoryType] = None,
    limit: int = 100,
    skip: int = 0
) -> List[models.MemoryEntry]:
    """지정된 조건에 맞는 메모리 항목 목록을 비동기적으로 조회합니다."""
    logger.debug(f"Getting memory entries for user {user_id} with filters: session_id={session_id}, scope={scope}, type={memory_type}, limit={limit}, skip={skip}")
    stmt = select(models.MemoryEntry).filter(models.MemoryEntry.user_id == user_id)

    if session_id:
        if scope == schemas.MemoryScope.SESSION:
            stmt = stmt.filter(models.MemoryEntry.session_id == session_id)
        elif scope == schemas.MemoryScope.SHARED:
            stmt = stmt.filter(models.MemoryEntry.scope == schemas.MemoryScope.SHARED.value)
        else:
            stmt = stmt.filter(
                (models.MemoryEntry.session_id == session_id) |
                (models.MemoryEntry.scope == schemas.MemoryScope.SHARED.value)
            )
    else:
        if scope == schemas.MemoryScope.SESSION:
            logger.warning("Requesting Session scope memories without providing session_id.")
            return []
        elif scope == schemas.MemoryScope.SHARED:
            stmt = stmt.filter(models.MemoryEntry.scope == schemas.MemoryScope.SHARED.value)


    if memory_type:
        stmt = stmt.filter(models.MemoryEntry.memory_type == memory_type.value)

    stmt = stmt.order_by(models.MemoryEntry.timestamp.desc())
    if skip > 0:
        stmt = stmt.offset(skip)
    if limit > 0:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()

async def get_memory_entry(db: AsyncSession, memory_id: int, user_id: int) -> Optional[models.MemoryEntry]:
    """특정 ID의 메모리 항목을 비동기적으로 조회합니다. (소유권 확인 포함)"""
    logger.debug(f"Querying memory entry with ID: {memory_id} for user ID: {user_id}")
    stmt = select(models.MemoryEntry).filter(models.MemoryEntry.id == memory_id, models.MemoryEntry.user_id == user_id)
    result = await db.execute(stmt)
    memory = result.scalars().first()
    if memory:
        logger.debug(f"Memory entry found: {memory.id}")
    else:
        logger.warning(f"Memory entry {memory_id} not found or not owned by user {user_id}")
    return memory

async def update_memory_entry(
    db: AsyncSession,
    memory_id: int,
    memory_update: schemas.MemoryEntryUpdate,
    user_id: int
) -> Optional[models.MemoryEntry]:
    """기존 메모리 항목을 비동기적으로 수정합니다."""
    db_memory = await get_memory_entry(db, memory_id=memory_id, user_id=user_id)
    if db_memory:
        update_data = memory_update.model_dump(exclude_unset=True)
        if not update_data:
             return db_memory

        logger.debug(f"Updating memory entry {memory_id} with data: {update_data}")
        for key, value in update_data.items():
            if isinstance(value, Enum):
                setattr(db_memory, key, value.value)
            elif hasattr(db_memory, key):
                setattr(db_memory, key, value)
            else:
                logger.warning(f"MemoryEntry update attempt with invalid field: {key}")

        try:
            await db.commit()
            await db.refresh(db_memory)
            logger.info(f"Memory entry {memory_id} updated for user {user_id}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to update memory entry {memory_id} for user {user_id}: {e}", exc_info=True)
            raise
    return db_memory

async def delete_memory_entry(
    db: AsyncSession,
    memory_id: int,
    user_id: int
) -> Optional[models.MemoryEntry]:
    """메모리 항목을 비동기적으로 삭제합니다."""
    db_memory = await get_memory_entry(db, memory_id=memory_id, user_id=user_id)
    if db_memory:
        try:
            await db.delete(db_memory)
            await db.commit()
            logger.info(f"Memory entry {memory_id} deleted for user {user_id}")
            return db_memory
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to delete memory entry {memory_id}: {e}", exc_info=True)
            raise
    return None



from fastapi import FastAPI, Depends, HTTPException, status, Query, File, UploadFile, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from datetime import timedelta, timezone, datetime, date
import time
import uuid
import os
import shutil
import mimetypes
import logging
import logging.config
import asyncio

from . import models, schemas, crud, security
from .database import get_db
from .llm_clients import google_client, openai_client, anthropic_client
from .logging_config import LOGGING_CONFIG
from .mcp_client import MCPClient

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Rini API",
    description="LLM Wrapper API for Gemini, OpenAI, Claude.",
    version="0.1.0",
)

UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

mcp_client = MCPClient()



@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to Rini API!"}



@app.post("/users/", response_model=schemas.UserWithToken, status_code=status.HTTP_201_CREATED)
async def create_new_simple_user_and_get_token(db: AsyncSession = Depends(get_db)):
    logger.info("Attempting to create a new simple user and get token.")
    try:
        created_user = await crud.create_user(db=db)
        logger.info(f"User created with ID: {created_user.id}")

        access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            data={"sub": str(created_user.id)}, expires_delta=access_token_expires
        )
        logger.debug(f"Access token generated for user ID: {created_user.id}")

        response_data = {
            "id": created_user.id,
            "access_token": access_token,
            "token_type": "bearer"
        }
        logger.info(f"Successfully created user and token for user ID: {created_user.id}")
        return response_data
    except Exception as e:
        logger.exception(f"Error creating new simple user and token: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating user and token.")

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(
    current_user_id: int = Depends(security.get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Attempting to read user info for user ID: {current_user_id}")
    try:
        user = await crud.get_user(db, user_id=current_user_id)
        if user is None:
            logger.warning(f"User not found for ID: {current_user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        logger.info(f"Successfully retrieved user info for user ID: {current_user_id}")
        return user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading user info for user ID {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading user info.")


@app.post("/api-keys/", response_model=schemas.ApiKey, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(
    api_key_in: schemas.ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to create API key for provider: {api_key_in.model_provider}")
    try:
        created_api_key = await crud.create_api_key(db=db, api_key=api_key_in, user_id=current_user_id)
        logger.info(f"API key created successfully for user {current_user_id}, provider: {api_key_in.model_provider}, API Key ID: {created_api_key.id}")
        return created_api_key
    except HTTPException as e:
        logger.error(f"HTTPException while creating API key for user {current_user_id}, provider {api_key_in.model_provider}: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"Error creating API key for user {current_user_id}, provider {api_key_in.model_provider}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating API key.")

@app.get("/api-keys/", response_model=List[schemas.ApiKey])
async def read_user_api_keys(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read API keys with skip: {skip}, limit: {limit}")
    try:
        api_keys = await crud.get_api_keys_by_user(db=db, user_id=current_user_id, skip=skip, limit=limit)
        logger.info(f"Successfully retrieved {len(api_keys)} API keys for user {current_user_id}")
        return api_keys
    except Exception as e:
        logger.exception(f"Error reading API keys for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading API keys.")

@app.get("/api-keys/{api_key_id}", response_model=schemas.ApiKey)
async def read_single_api_key(
    api_key_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read API key ID: {api_key_id}")
    try:
        db_api_key = await crud.get_api_key(db, api_key_id=api_key_id, user_id=current_user_id)
        if db_api_key is None:
            logger.warning(f"API Key ID {api_key_id} not found or not authorized for user {current_user_id}")
            raise HTTPException(status_code=404, detail="API Key not found or not authorized")
        logger.info(f"Successfully retrieved API key ID {api_key_id} for user {current_user_id}")
        return db_api_key
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading API key ID {api_key_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading API key.")

@app.put("/api-keys/{api_key_id}", response_model=schemas.ApiKey)
async def update_user_api_key(
    api_key_id: int,
    api_key_update: schemas.ApiKeyUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to update API key ID: {api_key_id}")
    try:
        updated_api_key = await crud.update_api_key(db, api_key_id=api_key_id, api_key_update=api_key_update, user_id=current_user_id)
        if updated_api_key is None:
            logger.warning(f"API Key ID {api_key_id} not found or not authorized for update by user {current_user_id}")
            raise HTTPException(status_code=404, detail="API Key not found or not authorized to update")
        logger.info(f"Successfully updated API key ID {api_key_id} for user {current_user_id}")
        return updated_api_key
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error updating API key ID {api_key_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while updating API key.")

@app.delete("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_api_key(
    api_key_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to delete API key ID: {api_key_id}")
    try:
        deleted_api_key = await crud.delete_api_key(db, api_key_id=api_key_id, user_id=current_user_id)
        if deleted_api_key is None:
            logger.warning(f"API Key ID {api_key_id} not found or not authorized for deletion by user {current_user_id}")
            raise HTTPException(status_code=404, detail="API Key not found or not authorized to delete")
        logger.info(f"Successfully deleted API key ID {api_key_id} for user {current_user_id}")
        return
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error deleting API key ID {api_key_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while deleting API key.")


@app.post("/sessions/", response_model=schemas.Session, status_code=status.HTTP_201_CREATED)
async def create_new_session(
    session_in: schemas.SessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to create a new session with name: {session_in.alias}")
    try:
        created_session = await crud.create_session(db=db, session=session_in, user_id=current_user_id)
        logger.info(f"Session created successfully for user {current_user_id} with ID: {created_session.id}")
        return created_session
    except Exception as e:
        logger.exception(f"Error creating session for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating session.")

@app.get("/sessions/", response_model=List[schemas.Session])
async def read_user_sessions(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read sessions with skip: {skip}, limit: {limit}")
    try:
        sessions = await crud.get_sessions_by_user(db=db, user_id=current_user_id, skip=skip, limit=limit)
        logger.info(f"Successfully retrieved {len(sessions)} sessions for user {current_user_id}")
        return sessions
    except Exception as e:
        logger.exception(f"Error reading sessions for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading sessions.")

@app.get("/sessions/{session_id}", response_model=schemas.Session)
async def read_single_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read session ID: {session_id}")
    try:
        db_session = await crud.get_session(db, session_id=session_id, user_id=current_user_id)
        if db_session is None:
            logger.warning(f"Session ID {session_id} not found or not authorized for user {current_user_id}")
            raise HTTPException(status_code=404, detail="Session not found or not authorized")
        logger.info(f"Successfully retrieved session ID {session_id} for user {current_user_id}")
        return db_session
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading session ID {session_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading session.")

@app.put("/sessions/{session_id}", response_model=schemas.Session)
async def update_user_session(
    session_id: str,
    session_update: schemas.SessionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to update session ID: {session_id}")
    try:
        updated_session = await crud.update_session(db, session_id=session_id, session_update=session_update, user_id=current_user_id)
        if updated_session is None:
            logger.warning(f"Session ID {session_id} not found or not authorized for update by user {current_user_id}")
            raise HTTPException(status_code=404, detail="Session not found or not authorized to update")
        logger.info(f"Successfully updated session ID {session_id} for user {current_user_id}")
        return updated_session
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error updating session ID {session_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while updating session.")

@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to delete session ID: {session_id}")
    try:
        deleted_session = await crud.delete_session(db, session_id=session_id, user_id=current_user_id)
        if deleted_session is None:
            logger.warning(f"Session ID {session_id} not found or not authorized for deletion by user {current_user_id}")
            raise HTTPException(status_code=404, detail="Session not found or not authorized to delete")
        logger.info(f"Successfully deleted session ID {session_id} for user {current_user_id}")
        return
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error deleting session ID {session_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while deleting session.")


@app.post("/sessions/{session_id}/messages/", response_model=schemas.Message, status_code=status.HTTP_201_CREATED)
async def create_new_message_for_session(
    session_id: str,
    message_in: schemas.MessageCreateBase,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to create message in session {session_id} with role {message_in.role}")
    try:
        db_session = await crud.get_session(db, session_id=session_id, user_id=current_user_id)
        if not db_session:
            logger.warning(f"Session {session_id} not found or not authorized for user {current_user_id}")
            raise HTTPException(status_code=404, detail="Session not found or not authorized")
        logger.debug(f"Session {session_id} verified for user {current_user_id}")

        message_data_internal = schemas.MessageCreateInternal(
            role=message_in.role,
            content=message_in.content,
            session_id=session_id,
            user_id=current_user_id
        )
        logger.debug(f"Prepared message data for session {session_id}")

        created_message = await crud.create_message(db=db, message_data=message_data_internal, requesting_user_id=current_user_id)
        logger.info(f"Message created successfully in session {session_id} with ID: {created_message.id}")
        return created_message
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error creating message in session {session_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating message.")

@app.get("/sessions/{session_id}/messages/", response_model=List[schemas.Message])
async def read_messages_in_session(
    session_id: str,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read messages in session {session_id} with skip: {skip}, limit: {limit}")
    try:
        messages = await crud.get_messages_by_session(db, session_id=session_id, user_id=current_user_id, skip=skip, limit=limit)
        if not messages and not await crud.get_session(db, session_id=session_id, user_id=current_user_id):
            logger.warning(f"Session {session_id} not found or not authorized for user {current_user_id} when reading messages.")
            raise HTTPException(status_code=404, detail="Session not found or not authorized")
        logger.info(f"Successfully retrieved {len(messages)} messages from session {session_id} for user {current_user_id}")
        return messages
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading messages from session {session_id} for user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading messages.")

@app.get("/messages/{message_id}", response_model=schemas.Message)
async def read_message_info(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    logger.info(f"User {current_user_id} attempting to read message info for message ID: {message_id}")
    try:
        db_message = await crud.get_message(db, message_id=message_id, user_id=current_user_id)
        if db_message is None:
            logger.warning(f"Message ID {message_id} not found or not authorized for user {current_user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found or you do not have permission to access it."
            )
        logger.info(f"Successfully retrieved message info for message ID {message_id}, user {current_user_id}")
        return db_message
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading message info for message ID {message_id}, user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading message info.")

@app.get("/messages/{message_id}/history", response_model=List[schemas.Message])
async def read_message_parent_history(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    """
    특정 메시지 ID로부터 시작하여 루트 메시지까지 거슬러 올라가는
    전체 부모 메시지 체인(요청 메시지 포함)을 시간 순서대로 반환합니다.
    요청한 사용자가 해당 메시지 체인이 속한 세션의 소유주일 경우에만 반환됩니다.
    """
    logger.info(f"User {current_user_id} attempting to read message history starting from message ID: {message_id}")
    try:
        message_history = await crud.get_message_parent_history(db, message_id=message_id, user_id=current_user_id)
        if not message_history:
            original_message = await crud.get_message(db, message_id=message_id, user_id=current_user_id)
            if not original_message:
                 logger.warning(f"Message ID {message_id} not found or not authorized for user {current_user_id} when reading history.")
                 raise HTTPException(
                     status_code=status.HTTP_404_NOT_FOUND,
                     detail="Message not found or you do not have permission to access this message chain."
                 )
            logger.info(f"Message history is empty (likely root message) for message ID {message_id}, user {current_user_id}")
        logger.info(f"Successfully retrieved {len(message_history)} messages in history for message ID {message_id}, user {current_user_id}")
        return message_history
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error reading message history for message ID {message_id}, user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading message history.")


@app.post("/llm/text-completion/", response_model=schemas.TextCompletionResponse)
async def get_text_completion(
    request_data: schemas.TextCompletionRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    """
    사용자 텍스트 입력을 받아 지정된 LLM 제공사/모델을 호출하고 텍스트 응답을 반환합니다.
    (현재 버전은 세션 저장/로드 기능을 포함하지 않습니다.)
    """
    provider = request_data.provider.value
    model_name = request_data.model
    input_text = request_data.text
    session_id = request_data.session_id
    llm_params = request_data.llm_params if request_data.llm_params else {}

    logger.info(f"User {current_user_id} requesting text completion. Provider: {provider}, Model: {model_name}, Session: {session_id}")
    logger.debug(f"LLM Params: {llm_params}, Input Text Length: {len(input_text)}")

    db_session: Optional[models.Session] = None
    message_history_for_llm: List[Dict[str, Any]] = []
    system_prompt: Optional[str] = None
    memory_prompt_part: Optional[str] = None

    if session_id:
        logger.debug(f"Session ID {session_id} provided, attempting to retrieve session.")
        db_session = await crud.get_session(db, session_id=session_id, user_id=current_user_id)
        if not db_session:
            logger.warning(f"Session {session_id} not found or not authorized for user {current_user_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or not authorized.")
        logger.info(f"Session {session_id} retrieved successfully.")

        if db_session.memory_mode != schemas.MemoryMode.DISABLED:
            relevant_memories = await crud.get_memory_entries(
                db, user_id=current_user_id, session_id=session_id, limit=1000
            )
            if relevant_memories:
                memory_items_str = [f"[{mem.memory_type}/{mem.scope}] {mem.content}" for mem in reversed(relevant_memories)]
                memory_prompt_part = "[알고 있어야 할 메모리 내용]\n" + "\n".join(f"- {item}" for item in memory_items_str) + "\n\n"
                logger.info(f"Injecting {len(relevant_memories)} memory entries into prompt for session {session_id}")

        system_prompt = db_session.system_prompt
        if system_prompt:
            message_history_for_llm.append({"role": "system", "content": system_prompt})
        if memory_prompt_part:
            message_history_for_llm.append({"role": "user", "content": memory_prompt_part})
        
        history_messages = await crud.get_messages_by_session(db, session_id=session_id, user_id=current_user_id, limit=None)
        for msg in history_messages:
            message_history_for_llm.append({"role": msg.role.value, "content": msg.content})

    message_history_for_llm.append({"role": "user", "content": input_text})
    final_messages_for_llm = [{"role": msg["role"], "content": msg.get("content")} for msg in message_history_for_llm]

    db_api_key = await crud.get_api_key_by_provider(db, user_id=current_user_id, provider=provider)
    if not db_api_key:
        logger.error(f"API key for provider '{provider}' not found for user {current_user_id}.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"API key for provider '{provider}' not found.")
    try:
        decrypted_api_key = security.decrypt_api_key(db_api_key._encrypted_api_key)
    except ValueError as e:
        logger.exception(f"API Key decryption failed for key id {db_api_key.id}, user {current_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not process API key.")

    user_db_message: Optional[models.Message] = None
    if db_session:
        last_message = await crud.get_last_message_in_session(db, session_id=session_id)
        parent_id_for_user_msg = last_message.id if last_message else None
        new_user_message_model = models.Message(
            session_id=session_id,
            parent_message_id=parent_id_for_user_msg,
            user_id=current_user_id,
            role=schemas.MessageRole.USER,
            content=input_text,
            created_at=datetime.now(timezone.utc)
        )
        db.add(new_user_message_model)
        try:
            await db.commit()
            await db.refresh(new_user_message_model)
            user_db_message = new_user_message_model
        except Exception as e:
            await db.rollback()
            logger.exception(f"Failed to save user message for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save user message: {e}")

    llm_response_data: Dict[str, Any] = {}
    response_text = ""
    start_time = time.monotonic()
    try:
        if provider == schemas.LLMProvider.GOOGLE.value:
            llm_response_data = await google_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=final_messages_for_llm,
                model=model_name,
                mcp_client=mcp_client,
                **llm_params
            )
        elif provider == schemas.LLMProvider.OPENAI.value:
            llm_response_data = await openai_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=final_messages_for_llm, 
                model=model_name,
                mcp_client=mcp_client,
                **llm_params
            )
        elif provider == schemas.LLMProvider.ANTHROPIC.value:
            user_assistant_messages = [m for m in final_messages_for_llm if m['role'] != 'system']
            llm_response_data = await anthropic_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=user_assistant_messages,
                model=model_name,
                system_prompt=system_prompt,
                mcp_client=mcp_client,
                **llm_params
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
        response_text = llm_response_data.get("text", "")
    except (ConnectionError, ConnectionAbortedError, ValueError, RuntimeError) as e:

        status_code_val = status.HTTP_503_SERVICE_UNAVAILABLE
        detail_val = f"LLM service error: {e}"
        if isinstance(e, ValueError):
            status_code_val = status.HTTP_400_BAD_REQUEST
            detail_val = f"LLM request error: {e}"
        elif isinstance(e, ConnectionAbortedError):
             status_code_val = status.HTTP_429_TOO_MANY_REQUESTS
             detail_val = f"LLM rate limit exceeded: {e}"
        logger.error(f"LLM call failed. Provider: {provider}, Model: {model_name}. Status: {status_code_val}, Detail: {detail_val}", exc_info=True)
        raise HTTPException(status_code=status_code_val, detail=detail_val)
    except Exception as e:
        logger.exception(f"Unexpected error during LLM call. Provider: {provider}, Model: {model_name}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during LLM call.")
    finally:
        end_time = time.monotonic()
        response_time_ms = int((end_time - start_time) * 1000)
        logger.info(f"LLM call ({provider}/{model_name}) processing time: {response_time_ms} ms")

    assistant_db_message: Optional[models.Message] = None
    if db_session:
        parent_id_for_assistant_msg = user_db_message.id if user_db_message else None
        prompt_tokens = llm_response_data.get("prompt_tokens")
        completion_tokens = llm_response_data.get("completion_tokens")
        new_assistant_message_model = models.Message(
            session_id=session_id,
            parent_message_id=parent_id_for_assistant_msg,
            user_id=current_user_id,
            role=schemas.MessageRole.ASSISTANT,
            content=response_text,
            model_used=model_name,
            response_time_ms=response_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            created_at=datetime.now(timezone.utc)
        )
        db.add(new_assistant_message_model)
        try:
            await db.commit()
            await db.refresh(new_assistant_message_model)
            assistant_db_message = new_assistant_message_model
        except Exception as e:
            await db.rollback()
            logger.exception(f"Failed to save assistant message for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save assistant message: {e}")

    return schemas.TextCompletionResponse(
        response_text=response_text, session_id=session_id,
        user_message=schemas.Message.model_validate(user_db_message) if user_db_message else None,
        assistant_message=schemas.Message.model_validate(assistant_db_message) if assistant_db_message else None
    )

@app.post("/llm/chat-completions/", response_model=schemas.MessagesCompletionResponse)
async def get_stateless_chat_completion(
    request_data: schemas.MessagesCompletionRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    """
    메시지 목록을 입력받아 지정된 LLM을 호출하고,
    세션에 대화 내용을 저장한 후 어시스턴트의 응답 메시지를 반환합니다.
    (주의: 이 엔드포인트는 상태 비저장(stateless)이므로 세션 저장을 수행하지 않습니다.)
    """
    provider = request_data.provider.value
    model_name = request_data.model
    input_messages_data = request_data.messages
    llm_params = request_data.llm_params if request_data.llm_params else {}

    db_api_key = await crud.get_api_key_by_provider(db, user_id=current_user_id, provider=provider)
    if not db_api_key:
        raise HTTPException(status_code=400, detail=f"API key for provider '{provider}' not found.")
    try:
        decrypted_api_key = security.decrypt_api_key(db_api_key._encrypted_api_key)
    except ValueError as e:
        raise HTTPException(status_code=500, detail="Could not process API key.")

    messages_for_llm = [msg.model_dump(include={'role', 'content'}) for msg in input_messages_data]
    system_prompt_in_messages: Optional[str] = None
    user_assistant_messages_for_llm = messages_for_llm
    if messages_for_llm and messages_for_llm[0]['role'] == 'system':
        system_prompt_in_messages = messages_for_llm[0]['content']
        user_assistant_messages_for_llm = messages_for_llm[1:]

    llm_response_data: Dict[str, Any] = {}
    start_time = time.monotonic()
    response_text = ""
    prompt_tokens = None
    completion_tokens = None
    try:
        if provider == schemas.LLMProvider.GOOGLE.value:
            llm_response_data = await google_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=messages_for_llm,
                model=model_name,
                mcp_client=mcp_client,
                **llm_params
            )
        elif provider == schemas.LLMProvider.OPENAI.value:
            llm_response_data = await openai_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=messages_for_llm,
                model=model_name,
                mcp_client=mcp_client,
                **llm_params
            )
        elif provider == schemas.LLMProvider.ANTHROPIC.value:
            llm_response_data = await anthropic_client.generate_text_from_messages_async(
                api_key=decrypted_api_key,
                messages=user_assistant_messages_for_llm,
                model=model_name,
                system_prompt=system_prompt_in_messages,
                mcp_client=mcp_client,
                **llm_params
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
        response_text = llm_response_data.get("text", "")
        prompt_tokens = llm_response_data.get("prompt_tokens")
        completion_tokens = llm_response_data.get("completion_tokens")
    except (ConnectionError, ConnectionAbortedError, ValueError, RuntimeError) as e:

        status_code_val = status.HTTP_503_SERVICE_UNAVAILABLE
        detail_val = f"LLM service error: {e}"
        if isinstance(e, ValueError):
            status_code_val = status.HTTP_400_BAD_REQUEST
            detail_val = f"LLM request error: {e}"
        elif isinstance(e, ConnectionAbortedError):
             status_code_val = status.HTTP_429_TOO_MANY_REQUESTS
             detail_val = f"LLM rate limit exceeded: {e}"
        logger.error(f"Stateless chat completion failed. Provider: {provider}, Model: {model_name}. Status: {status_code_val}, Detail: {detail_val}", exc_info=True)
        raise HTTPException(status_code=status_code_val, detail=detail_val)
    except Exception as e:
        logger.exception(f"Unexpected error during stateless chat completion. Provider: {provider}, Model: {model_name}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during stateless chat completion.")
    finally:
        end_time = time.monotonic()
        response_time_ms = int((end_time - start_time) * 1000)
        logger.info(f"Stateless chat completion ({provider}/{model_name}) processing time: {response_time_ms} ms")
    
    return schemas.MessagesCompletionResponse(
        response_text=response_text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


@app.post("/llm/embeddings/", response_model=schemas.EmbeddingResponse)
async def create_embedding(
    request_data: schemas.EmbeddingRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    """
    주어진 텍스트 또는 텍스트 목록에 대한 임베딩 벡터를 생성합니다.
    """
    provider = request_data.provider.value
    model_name = request_data.model
    input_data = request_data.input
    input_type = "string" if isinstance(input_data, str) else "list"
    input_count = 1 if input_type == "string" else len(input_data)

    db_api_key = await crud.get_api_key_by_provider(db, user_id=current_user_id, provider=provider)
    if not db_api_key:
        raise HTTPException(status_code=400, detail=f"API key for provider '{provider}' not found.")
    try:
        decrypted_api_key = security.decrypt_api_key(db_api_key._encrypted_api_key)
    except ValueError as e:
        raise HTTPException(status_code=500, detail="Could not process API key.")

    embedding_response_data: Dict[str, Any] = {}
    start_time = time.monotonic()
    try:
        if provider == schemas.LLMProvider.OPENAI.value:
            if input_type == "string":
                embedding_response_data = await openai_client.get_embedding_async(
                    api_key=decrypted_api_key,
                    text=input_data,
                    model=model_name
                )
            elif input_type == "list":
                embedding_response_data = await openai_client.get_embedding_batch_async(
                    api_key=decrypted_api_key,
                    texts=input_data,
                    model=model_name
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid input type.")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding provider: {provider}")
    except (ConnectionError, ConnectionAbortedError, ValueError, RuntimeError) as e:

        status_code_val = status.HTTP_503_SERVICE_UNAVAILABLE
        detail_val = f"Embedding service error: {e}"
        if isinstance(e, ValueError):
            status_code_val = status.HTTP_400_BAD_REQUEST
            detail_val = f"Embedding request error: {e}"
        elif isinstance(e, ConnectionAbortedError):
             status_code_val = status.HTTP_429_TOO_MANY_REQUESTS
             detail_val = f"Embedding rate limit exceeded: {e}"
        logger.error(f"Embedding call failed. Provider: {provider}, Model: {model_name}. Status: {status_code_val}, Detail: {detail_val}", exc_info=True)
        raise HTTPException(status_code=status_code_val, detail=detail_val)
    except Exception as e:
        logger.exception(f"Unexpected error during embedding generation. Provider: {provider}, Model: {model_name}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during embedding generation.")
    finally:
        end_time = time.monotonic()
        response_time_ms = int((end_time - start_time) * 1000)
        logger.info(f"Embedding call ({provider}/{model_name}) processing time: {response_time_ms} ms")
        
    return schemas.EmbeddingResponse(**embedding_response_data)

@app.post("/llm/image-completion/", response_model=schemas.ImageCompletionResponse)
async def handle_image_completion(
    image_file: UploadFile = File(..., description="분석할 이미지 파일"),
    prompt: Optional[str] = Form(None),
    provider: schemas.LLMProvider = Form(...),
    model: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    """
    이미지 파일과 선택적 텍스트 프롬프트를 받아 LLM을 호출하고 텍스트 응답을 반환합니다.
    session_id가 제공되면 해당 세션에 요청(이미지 참조 포함)과 응답 메시지를 저장합니다.
    """
    logger.info(f"User {current_user_id} requesting image completion. Provider: {provider.value}, Model: {model}, Session: {session_id}")
    saved_file_path: Optional[str] = None
    model_to_use = model

    if provider not in [schemas.LLMProvider.GOOGLE, schemas.LLMProvider.OPENAI]:
        raise HTTPException(status_code=400, detail="Unsupported provider for image completion.")

    try:
        original_filename = image_file.filename
        file_extension = os.path.splitext(original_filename)[1] if original_filename else ".tmp"
        saved_filename = f"{uuid.uuid4()}{file_extension}"
        saved_file_path = os.path.join(UPLOAD_DIRECTORY, saved_filename)
        detected_mime_type, _ = mimetypes.guess_type(original_filename)
        detected_mime_type = detected_mime_type or image_file.content_type or 'application/octet-stream'

        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        logger.info(f"File saved: {saved_file_path}")

        db_session: Optional[models.Session] = None
        if session_id:
            db_session = await crud.get_session(db, session_id=session_id, user_id=current_user_id)
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found or not authorized.")

        db_api_key = await crud.get_api_key_by_provider(db, user_id=current_user_id, provider=provider.value)
        if not db_api_key:
            raise HTTPException(status_code=400, detail=f"API key for '{provider.value}' not found.")
        decrypted_api_key = security.decrypt_api_key(db_api_key._encrypted_api_key)

        user_db_message: Optional[models.Message] = None
        if db_session:
            last_message = await crud.get_last_message_in_session(db, session_id=session_id)
            parent_id_for_user_msg = last_message.id if last_message else None
            new_user_message_model = models.Message(
                session_id=session_id,
                parent_message_id=parent_id_for_user_msg,
                user_id=current_user_id,
                role=schemas.MessageRole.USER,
                content=prompt,
                file_reference=saved_file_path,
                original_filename=original_filename,
                mime_type=detected_mime_type,
                created_at=datetime.now(timezone.utc)
            )
            db.add(new_user_message_model)
            await db.commit(); await db.refresh(new_user_message_model)
            user_db_message = new_user_message_model

        response_text = ""; start_time = time.monotonic(); llm_response_data = {}
        prompt_tokens = None; completion_tokens = None
        
        try:
            if provider == schemas.LLMProvider.GOOGLE:
                if not model_to_use: model_to_use = "gemini-1.5-flash"
                llm_response_data = await google_client.generate_text_from_image_and_text_async(
                    api_key=decrypted_api_key,
                    image_path=saved_file_path,
                    prompt=prompt or "",
                    mcp_client=mcp_client,

                )
            elif provider == schemas.LLMProvider.OPENAI:
                if not model_to_use: model_to_use = "gpt-4o"

                with open(saved_file_path, "rb") as f: image_bytes = f.read()
                llm_response_data = await openai_client.generate_text_from_image_and_text_async(
                    api_key=decrypted_api_key,
                    image_bytes=image_bytes,
                    mime_type=detected_mime_type,
                    prompt=prompt,
                    model=model_to_use,
                    mcp_client=mcp_client,
                )
            response_text = llm_response_data.get("text", "")
            prompt_tokens = llm_response_data.get("prompt_tokens")
            completion_tokens = llm_response_data.get("completion_tokens")
        except (ConnectionError, ConnectionAbortedError, ValueError, RuntimeError) as e_llm:

            status_code_llm = status.HTTP_503_SERVICE_UNAVAILABLE
            detail_llm = f"LLM service error: {e_llm}"
            if isinstance(e_llm, ValueError) and "API key" in str(e_llm): status_code_llm = 401; detail_llm = str(e_llm)
            elif isinstance(e_llm, ConnectionAbortedError): status_code_llm = 429; detail_llm = f"LLM rate limit exceeded: {e_llm}"
            elif isinstance(e_llm, RuntimeError) and "LLM 처리 오류" in str(e_llm): status_code_llm = 500; detail_llm = str(e_llm)
            logger.error(f"Image completion call failed. Model: {model_to_use}. Status: {status_code_llm}, Detail: {detail_llm}", exc_info=True)
            raise HTTPException(status_code=status_code_llm, detail=detail_llm)
        finally:
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            logger.info(f"Image completion call ({provider.value}/{model_to_use}) processing time: {response_time_ms} ms")

        assistant_db_message: Optional[models.Message] = None
        if db_session:
            parent_id_for_assistant_msg = user_db_message.id if user_db_message else None
            new_assistant_message_model = models.Message(
                session_id=session_id,
                parent_message_id=parent_id_for_assistant_msg,
                user_id=current_user_id,
                role=schemas.MessageRole.ASSISTANT,
                content=response_text,
                model_used=model_to_use,
                response_time_ms=response_time_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                created_at=datetime.now(timezone.utc)
            )
            db.add(new_assistant_message_model)
            await db.commit(); await db.refresh(new_assistant_message_model)
            assistant_db_message = new_assistant_message_model
        
        return schemas.ImageCompletionResponse(
            response_text=response_text,
            assistant_message=schemas.Message.model_validate(assistant_db_message) if assistant_db_message else None,
            session_id=session_id
        )
    except Exception as e:
        logger.exception(f"Error in handle_image_completion: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if saved_file_path and os.path.exists(saved_file_path):
            try: os.remove(saved_file_path)
            except Exception as e_rem: logger.warning(f"Failed to remove temp file {saved_file_path}: {e_rem}")
        if image_file: await image_file.close()
    

@app.get("/usage/cost-estimation/", response_model=schemas.CostEstimationResponse)
async def get_cost_estimation(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    session_id: Optional[str] = Query(None),
    provider: Optional[schemas.LLMProvider] = Query(None),
    model: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    provider_value = provider.value if provider else None
    try:
        cost_data = await crud.calculate_estimated_cost(
            db=db,
            user_id=current_user_id,
            start_date=start_date,
            end_date=end_date,
            session_id=session_id,
            provider=provider_value,
            model=model
        )
        return cost_data
    except Exception as e:
        logger.exception(f"Error calculating cost estimation: {e}")
        raise HTTPException(status_code=500, detail="Error calculating cost.")


@app.post("/sessions/{session_id}/memory", response_model=schemas.MemoryEntry, status_code=status.HTTP_201_CREATED)
async def add_memory_entry(
    session_id: str,
    memory_data: schemas.MemoryEntryCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    db_session = await crud.get_session(db, session_id=session_id, user_id=current_user_id)
    if not db_session and memory_data.scope == schemas.MemoryScope.SESSION:
         raise HTTPException(status_code=404, detail="Session not found or not authorized.")
    try:
        created_memory = await crud.create_memory_entry(
            db=db,
            memory_data=memory_data,
            user_id=current_user_id,
            session_id=session_id
        )
        return created_memory
    except Exception as e:
        logger.exception(f"Failed to create memory entry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory entry: {e}")

@app.get("/memory/", response_model=List[schemas.MemoryEntry])
async def list_memory_entries(
    session_id: Optional[str] = Query(None),
    scope: Optional[schemas.MemoryScope] = Query(None),
    memory_type: Optional[schemas.MemoryType] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    memories = await crud.get_memory_entries(
        db,
        user_id=current_user_id,
        session_id=session_id,
        scope=scope,
        memory_type=memory_type,
        skip=skip,
        limit=limit
    )
    return memories

@app.get("/memory/{memory_id}", response_model=schemas.MemoryEntry)
async def get_single_memory_entry(
    memory_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    db_memory = await crud.get_memory_entry(db, memory_id=memory_id, user_id=current_user_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory entry not found or not authorized")
    return db_memory

@app.patch("/memory/{memory_id}", response_model=schemas.MemoryEntry)
async def update_single_memory_entry(
    memory_id: int,
    memory_update: schemas.MemoryEntryUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    updated_memory = await crud.update_memory_entry(db, memory_id=memory_id, memory_update=memory_update, user_id=current_user_id)
    if updated_memory is None:
        raise HTTPException(status_code=404, detail="Memory entry not found or not authorized to update")
    return updated_memory

@app.delete("/memory/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_memory_entry(
    memory_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(security.get_current_user_id)
):
    deleted_memory = await crud.delete_memory_entry(db, memory_id=memory_id, user_id=current_user_id)
    if deleted_memory is None:
        raise HTTPException(status_code=404, detail="Memory entry not found or not authorized to delete")
    return

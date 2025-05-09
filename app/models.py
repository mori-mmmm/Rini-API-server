

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Enum as DBEnum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base
from .schemas import MemoryMode, MessageRole

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)


    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


    api_keys = relationship("ApiKey", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="owner", cascade="all, delete-orphan")




class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_provider = Column(String, index=True, nullable=False)
    _encrypted_api_key = Column("encrypted_api_key", String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    owner = relationship("User", back_populates="api_keys")

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    alias = Column(String, index=True, nullable=True)
    system_prompt = Column(Text, nullable=True)
    memory_mode = Column(DBEnum(MemoryMode, name="memorymode_enum", create_constraint=True), default=MemoryMode.AUTO, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    owner = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    parent_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)


    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(DBEnum(MessageRole, name="messagerole_enum", create_constraint=True), nullable=False)
    content = Column(Text, nullable=True)
    file_reference = Column(String, nullable=True, index=True)
    original_filename = Column(String, nullable=True)
    mime_type = Column(String, nullable=True)
    model_used = Column(String, nullable=True)
    token_count = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="messages")



class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(Integer, primary_key=True, index=True)


    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    session_id = Column(String, ForeignKey("sessions.id"), nullable=True, index=True)


    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    memory_type = Column(String, nullable=False, index=True)
    scope = Column(String, nullable=False, default="Session", index=True)
    content = Column(Text, nullable=False)



    source_message_ids = Column(JSON, nullable=True)

    keywords = Column(JSON, nullable=True)

    importance = Column(String, nullable=True)


    owner = relationship("User")
    session = relationship("Session")
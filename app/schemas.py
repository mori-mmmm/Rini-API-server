

from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional, List, Dict, Union, Any
from enum import Enum


class MemoryMode(str, Enum):
    AUTO = "auto"
    EXPLICIT = "explicit"
    DISABLED = "disabled"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LLMProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class User(BaseModel):
    id: int

    class Config:
        from_attributes = True


class ApiKeyBase(BaseModel):
    model_provider: str = Field(..., description="LLM 제공사 이름 (예: 'openai', 'google', 'anthropic')", examples=["openai"])
    description: Optional[str] = Field(None, description="API 키에 대한 설명")

class ApiKeyCreate(ApiKeyBase):
    api_key_value: str = Field(..., description="사용자의 실제 LLM API 키 값 (서버에서 암호화 예정)")

class ApiKeyUpdate(BaseModel):
    description: Optional[str] = Field(None, description="API 키 설명 수정")

class ApiKey(ApiKeyBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class SessionBase(BaseModel):
    alias: Optional[str] = Field(None, description="사용자가 지정하는 세션 별칭", examples=["My Research Chat"])
    system_prompt: Optional[str] = Field(None, description="세션에 적용될 시스템 프롬프트")
    memory_mode: MemoryMode = Field(MemoryMode.AUTO, description="세션의 메모리 관리 방식")

class SessionCreate(SessionBase):
    pass

class SessionUpdate(SessionBase):
    pass

class Session(SessionBase):
    id: str
    user_id: int
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True



class MessageCreateBase(BaseModel):
    role: MessageRole = Field(..., description="메시지 발화자 역할")
    content: str = Field(..., description="텍스트 메시지 내용")



class MessageCreateInternal(MessageCreateBase):
    session_id: str
    user_id: int
    parent_message_id: Optional[int] = None


class Message(MessageCreateBase):
    id: int
    session_id: str
    user_id: int
    parent_message_id: Optional[int] = None

    file_reference: Optional[str] = Field(None, description="저장된 파일 참조(경로/ID 등)")
    original_filename: Optional[str] = Field(None, description="업로드된 원본 파일 이름")
    mime_type: Optional[str] = Field(None, description="파일 MIME 타입")


    model_used: Optional[str] = Field(None, description="응답 생성에 사용된 LLM 모델")
    token_count: Optional[int] = Field(None, description="해당 메시지(주로 assistant 응답)의 토큰 수")
    prompt_tokens: Optional[int] = Field(None, description="이 응답 생성에 사용된 프롬프트 토큰 수")
    completion_tokens: Optional[int] = Field(None, description="이 응답(assistant) 자체의 토큰 수")
    response_time_ms: Optional[int] = Field(None, description="LLM 응답 생성에 소요된 시간 (ms)")




    created_at: datetime

    class Config:
        from_attributes = True

class ImageCompletionResponse(BaseModel):
    response_text: str = Field(...)
    assistant_message: Optional[Message] = Field(None, description="저장된 어시스턴트 메시지 (세션 사용 시)")
    session_id: Optional[str] = Field(None)




class TextCompletionRequest(BaseModel):
    text: str = Field(..., description="사용자 입력 텍스트")
    session_id: Optional[str] = Field(None, description="기존 세션 ID (세션 이어가기 및 저장 시)")

    provider: LLMProvider = Field(..., description="사용할 LLM 제공사")
    model: str = Field(..., description="사용할 특정 모델 이름", examples=["gemini-2.5-pro-preview-05-06", "gpt-4o"])

    llm_params: Optional[Dict[str, Any]] = Field(None, description="LLM 특정 파라미터 (예: temperature, max_tokens)")

class TextCompletionResponse(BaseModel):
    response_text: str = Field(..., description="LLM의 텍스트 응답")
    session_id: Optional[str] = Field(None, description="응답이 포함된 (또는 새로 생성된) 세션 ID (세션 사용/생성 시)")
    user_message: Optional[Message] = Field(None, description="저장된 사용자 메시지 (세션 사용 시)")
    assistant_message: Optional[Message] = Field(None, description="저장된 어시스턴트 메시지 (세션 사용 시)")


class MessagesCompletionRequest(BaseModel):
    messages: List[MessageCreateBase]
    provider: LLMProvider = Field(..., description="사용할 LLM 제공사")
    model: str = Field(..., description="사용할 특정 모델 이름")
    llm_params: Optional[Dict[str, Any]] = Field(None, description="LLM 특정 파라미터 (예: temperature)")

class MessagesCompletionResponse(BaseModel):
    response_text: str = Field(..., description="LLM의 텍스트 응답")

    prompt_tokens: Optional[int] = Field(None, description="LLM 호출 시 사용된 프롬프트 토큰 수")
    completion_tokens: Optional[int] = Field(None, description="LLM이 생성한 완료 토큰 수")



class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserWithToken(User):
    access_token: str
    token_type: str = "bearer"



class EmbeddingRequest(BaseModel):


    input: Union[str, List[str]] = Field(..., description="임베딩을 생성할 텍스트 또는 텍스트 목록")

    provider: LLMProvider = Field(LLMProvider.OPENAI, description="사용할 임베딩 제공사")
    model: str = Field("text-embedding-3-large", description="사용할 임베딩 모델 이름")


class EmbeddingObject(BaseModel):
    embedding: List[float] = Field(..., description="생성된 임베딩 벡터")
    index: int = Field(0, description="입력에서의 인덱스 (단일 입력 시 0)")



class EmbeddingUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class EmbeddingResponse(BaseModel):

    data: List[EmbeddingObject] = Field(..., description="생성된 임베딩 객체 목록")
    model: str = Field(..., description="사용된 임베딩 모델 이름")
    usage: Optional[EmbeddingUsage] = Field(None, description="토큰 사용량 정보")



class CostEstimationDetail(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float


class CostEstimationResponse(BaseModel):
    total_estimated_cost: float = Field(..., description="조회 기간 동안의 총 예상 비용")
    currency: str = Field("USD", description="비용 통화 단위")
    start_date: Optional[date] = Field(None, description="조회 시작일")
    end_date: Optional[date] = Field(None, description="조회 종료일")
    details: Dict[str, CostEstimationDetail] = Field(..., description="모델별 상세 사용량 및 예상 비용")



class MemoryType(str, Enum):
    FACT = "Fact"
    RULE = "Rule"
    INSTRUCTION = "User Instruction"
    SUMMARY = "Summary"
    REFLECTION = "Model Reflection"
    TASK = "Task"
    USER_STATE = "User State"
    MODEL_STATE = "Model State"
    SYSTEM_CONFIG = "System Config"


class MemoryScope(str, Enum):
    SESSION = "Session"
    SHARED = "Shared"




class MemoryEntryBase(BaseModel):
    memory_type: MemoryType = Field(..., description="메모리의 종류")
    scope: MemoryScope = Field(MemoryScope.SESSION, description="메모리 적용 범위 (기본값: 세션)")
    content: str = Field(..., description="메모리의 실제 내용")
    source_message_ids: Optional[List[int]] = Field(None, description="메모리 생성과 관련된 메시지 ID 목록")
    keywords: Optional[List[str]] = Field(None, description="메모리 검색/분류를 위한 키워드 목록")
    importance: Optional[str] = Field(None, description="메모리 중요도 (예: High, Medium, Low)")


class MemoryEntryCreate(MemoryEntryBase):


    pass


class MemoryEntry(MemoryEntryBase):
    id: int
    user_id: int
    session_id: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True

class MemoryEntryUpdate(BaseModel):

    memory_type: Optional[MemoryType] = Field(None, description="메모리의 종류")
    scope: Optional[MemoryScope] = Field(None, description="메모리 적용 범위")
    content: Optional[str] = Field(None, description="메모리의 실제 내용")
    source_message_ids: Optional[List[int]] = Field(None, description="메모리 생성과 관련된 메시지 ID 목록")
    keywords: Optional[List[str]] = Field(None, description="메모리 검색/분류를 위한 키워드 목록")
    importance: Optional[str] = Field(None, description="메모리 중요도 (예: High, Medium, Low)")

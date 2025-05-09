# Rini API Server
Rini API는 다양한 대규모 언어 모델(LLM) 제공사(Google Gemini, OpenAI, Anthropic 등)의 API를 일관된 인터페이스를 통해 사용할 수 있도록 지원하는 비동기 FastAPI 기반 서비스입니다.  
사용자별 API 키 관리, 세션 관리, 대화 히스토리 저장 및 활용, 메시지별 사용량 추적, 외부 도구(MCP) 연동 설정, 사용자 주도 메모리 관리 등의 기능을 제공합니다.

## 주요 기능

### 🇦🇮 코어 LLM 래퍼
* **다중 LLM 제공사 지원:**
    * Google Gemini (텍스트 생성, 이미지 입력 처리)
    * OpenAI (텍스트 생성, 이미지 입력 처리, 텍스트 임베딩)
    * Anthropic Claude (텍스트 생성)
* **동적 모델 선택:** API 호출 시 사용할 LLM 제공사 및 모델을 유연하게 지정 가능.
* **LLM 추가 파라미터 전달:** `temperature`, `max_tokens` 등 LLM 호출 시 필요한 다양한 추가 인자를 `llm_params`를 통해 전달.
* **기본적인 LLM API 오류 처리:** 제공사 API 오류 발생 시 HTTP 에러로 변환하여 응답.

### 🔐 사용자 및 인증
* **간단한 사용자 시스템:** 시스템 내부 식별을 위한 `id` 기반의 `User` 모델.
* **사용자 생성 API:** `POST /users/` - 신규 사용자 생성.
* **JWT 기반 토큰 인증:**
    * `Authorization: Bearer <token>` 헤더를 통한 API 요청 인증.
    * 사용자 생성 시 액세스 토큰 함께 발급.
* **현재 사용자 정보 조회 API:** `GET /users/me` - 인증된 사용자의 기본 정보(ID) 조회.

### 🔑 API 키 관리
* **사용자별/제공사별 API 키 저장:** `model_provider` 기준으로 사용자 소유의 API 키 등록.
* **API 키 암호화:** AES-GCM 방식을 사용하여 데이터베이스에 안전하게 암호화하여 저장.
* **API 키 관리 엔드포인트:**
    * `POST /api-keys/`: 새 API 키 등록.
    * `GET /api-keys/`: 등록된 API 키 목록 조회.
    * `GET /api-keys/{api_key_id}`: 특정 API 키 상세 조회.
    * `PUT /api-keys/{api_key_id}`: API 키 정보(주로 설명) 수정.
    * `DELETE /api-keys/{api_key_id}`: API 키 삭제.

### 🧵 세션 관리
* **세션 생성 및 관리:** 대화 세션 생성, 조회, 설정 수정, 삭제 기능.
    * 세션별 `alias` (별칭), `system_prompt` (시스템 메시지), `memory_mode` (Enum: `AUTO`, `EXPLICIT`, `DISABLED`) 설정 가능(WIP).
    * UUID 기반 세션 ID 사용.
* **세션 관리 엔드포인트:**
    * `POST /sessions/`: 새 세션 생성.
    * `GET /sessions/`: 세션 목록 조회.
    * `GET /sessions/{session_id}`: 특정 세션 상세 조회.
    * `PUT /sessions/{session_id}`: 세션 정보 수정.
    * `DELETE /sessions/{session_id}`: 세션 삭제.

### 💬 메시지 및 대화 관리
* **메시지 영구 저장:** 역할(`user`, `assistant`, `system`), 내용(`content`), 파일 참조(`file_reference`, `original_filename`, `mime_type`) 등 메시지 상세 정보 데이터베이스 저장.
* **대화 흐름 추적:** `parent_message_id`를 통해 메시지 간의 순서 및 부모-자식 관계 저장.
* **LLM 메타데이터 저장:** 어시스턴트 메시지 생성 시 사용된 모델(`model_used`), 응답 시간(`response_time_ms`), 사용 토큰(`prompt_tokens`, `completion_tokens`) 기록.
* **세션 내 메시지 관리 엔드포인트:**
    * `POST /sessions/{session_id}/messages/`: 세션에 새 메시지 추가 (주로 사용자 메시지).
    * `GET /sessions/{session_id}/messages/`: 세션 내 메시지 목록 조회.
* **개별 메시지 조회 API:**
    * `GET /messages/{message_id}`: 특정 메시지 상세 정보 조회.
    * `GET /messages/{message_id}/history`: 특정 메시지부터 부모 메시지까지의 히스토리 체인 조회.

### 🧠 LLM 상호작용 API
* **텍스트 기반 완성 (`POST /llm/text-completion/`):**
    * 단일 텍스트 입력 지원.
    * `session_id` 유무에 따라 Stateless/Stateful 동작 분기:
        * Stateful 시: 대화 히스토리, 시스템 프롬프트, 메모리 내용을 함께 LLM에 전달 및 요청/응답 메시지 DB 저장.
        * Stateless 시: DB 저장 없이 일회성 호출 및 응답.
* **메시지 목록 기반 완성 (`POST /llm/chat-completions/`):**
    * `List[Message]` 형태의 메시지 목록 전체를 입력으로 받아 LLM 호출.
    * Stateless 동작 (DB 저장 없음, `session_id` 불필요).
* **이미지 입력 기반 완성 (`POST /llm/image-completion/`):**
    * 이미지 파일(`multipart/form-data`)과 선택적 텍스트 프롬프트 입력.
    * 현재 Google Gemini Vision, OpenAI Vision 모델 지원.
    * `session_id` 유무에 따라 Stateful/Stateless 동작 분기 (Stateful 시 사용자 메시지에 이미지 참조 정보 저장).
* **텍스트 임베딩 생성 (`POST /llm/embeddings/`):**
    * 단일 텍스트 또는 텍스트 목록을 입력받아 임베딩 벡터 생성.
    * 현재 OpenAI 임베딩 모델 지원.

### 🔧 외부 도구 (MCP) 연동 준비
* **현재 방식**
   * 서버 전체에 대한 MCP Client가 공유됨.
* **사용자별 MCP 서버 URL 등록/관리 API (WIP):**
    * `POST /mcp-connections/`: 새 MCP 서버 URL 정보 등록.
    * `GET /mcp-connections/`: 등록된 MCP 서버 URL 목록 조회.
    * `GET /mcp-connections/{connection_id}`: 특정 MCP 서버 URL 정보 상세 조회.
    * `PUT /mcp-connections/{connection_id}`: 정보 수정.
    * `DELETE /mcp-connections/{connection_id}`: 정보 삭제.
    * (실제 MCP 도구 호출 로직은 "미니턴 없이 한 턴에 하나의 function/MCP calling만 처리 후 결과 반환" 방식으로 설계 예정)

### 💾 메모리 관리 (사용자 주도)
* **세션별 메모리 모드(WIP):** `AUTO`, `EXPLICIT`, `DISABLED` 설정 가능.
* **메모리 항목 DB 저장:** `MemoryEntry` 테이블에 구조화된 메모리(타입, 범위, 내용, 관련 메시지 ID, 키워드, 중요도) 저장.
* **수동 메모리 관리 API:**
    * `POST /sessions/{session_id}/memory`: 새 메모리 항목 생성.
    * `GET /memory/`: 메모리 목록 조회 (필터링 지원: `session_id`, `scope`, `memory_type`).
    * `GET /memory/{memory_id}`: 특정 메모리 상세 조회.
    * `PATCH /memory/{memory_id}`: 특정 메모리 수정.
    * `DELETE /memory/{memory_id}`: 특정 메모리 삭제.
* **LLM 프롬프트 주입:** LLM 호출 시, 해당 세션의 메모리 모드가 `DISABLED`가 아니면 관련 메모리(세션 범위 + 공유 범위)를 조회하여 프롬프트에 자동으로 주입.

### 💰 사용량 및 비용 관리
* **예상 비용 계산 API (`GET /usage/cost-estimation/`):**
    * 데이터베이스에 저장된 어시스턴트 메시지의 `model_used`, `prompt_tokens`, `completion_tokens` 정보를 기반으로 예상 사용 비용 계산.
    * 기간, 세션 ID, 제공사, 모델별 필터링 지원.

## 설치 및 실행

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/mori-mmmm/Rini-API-server
    cd Rini-API-server
    ```
2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt 
    ```
4.  **`.env` 파일 설정:**
    * 프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 참고하여 실제 값으로 채웁니다:
        ```env
        RINI_API_MASTER_KEY= # AES 암호화를 위한 32바이트 랜덤 키 (base64 인코딩)
        JWT_SECRET_KEY= # JWT 서명을 위한 강력한 비밀 키
        JWT_ALGORITHM=HS256
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
        ```
5.  **데이터베이스 마이그레이션:**
    ```bash
    alembic upgrade head
    ```
6.  **서버 실행:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
7.  **API 문서:** 브라우저에서 `http://localhost:8000/docs` 또는 `http://localhost:8000/redoc` 접속.

## 향후 계획 (남은 작업 및 개선 사항)
* **메모리 기능 심화:** `AUTO` 모드 로직 (자동 요약/추출), `EXPLICIT` 모드 추가 API(예: 특정 대화 기반 요약 요청) 구현.
* **LLM 기능 확장:**
    * 다른 제공사(Gemini, Anthropic)의 이미지 입력 및 임베딩 기능 클라이언트/엔드포인트 구현.
    * 일반 파일 처리 기능 (RAG의 기반).
* **오류 처리 강화:** LLM 쿼터 오류 등 세분화된 오류 처리 및 표준화된 오류 응답 구조 적용.
* **로깅 상세 적용:** 애플리케이션 전반에 걸쳐 주요 이벤트 및 오류에 대한 로깅 강화.
* 대화 분기 기능
* 세션 히스토리 조회 (트리 형태)
* RAG (검색 증강 생성) - 파일 업로드 기반
* 에이전트(Agent) 기능

## 라이선스
MIT License

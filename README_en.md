# Rini API Server

Rini API is an asynchronous FastAPI-based service that allows you to use APIs from various Large Language Model (LLM) providers (such as Google Gemini, OpenAI, Anthropic, etc.) through a consistent interface.  
It provides features such as per-user API key management, session management, conversation history storage and utilization, per-message usage tracking, external tool (MCP) integration setup, and user-driven memory management.

## Key Features

### 🇦🇮 Core LLM Wrapper
* **Multi-LLM Provider Support:**
    * Google Gemini (Text generation, Image input processing)
    * OpenAI (Text generation, Image input processing, Text embedding)
    * Anthropic Claude (Text generation)
* **Dynamic Model Selection:** Flexible specification of LLM provider and model for each API call.
* **LLM Additional Parameter Passing:** Pass various additional arguments required for LLM calls, such as `temperature`, `max_tokens`, etc., through `llm_params`.
* **Basic LLM API Error Handling:** Converts provider API errors into HTTP errors in responses.

### 🔐 User and Authentication
* **Simple User System:** `User` model based on `id` for internal system identification.
* **User Creation API:** `POST /users/` - Creates a new user.
* **JWT-based Token Authentication:**
    * API request authentication via `Authorization: Bearer <token>` header.
    * Access token issued البروتوكول with user creation.
* **Current User Info API:** `GET /users/me` - Retrieves basic information (ID) of the authenticated user.

### 🔑 API Key Management
* **Per-User/Per-Provider API Key Storage:** Register user-owned API keys based on `model_provider`.
* **API Key Encryption:** Securely stores API keys in the database using AES-GCM encryption.
* **API Key Management Endpoints:**
    * `POST /api-keys/`: Register a new API key.
    * `GET /api-keys/`: Retrieve a list of registered API keys.
    * `GET /api-keys/{api_key_id}`: Retrieve details of a specific API key.
    * `PUT /api-keys/{api_key_id}`: Modify API key information (mainly description).
    * `DELETE /api-keys/{api_key_id}`: Delete an API key.

### 🧵 Session Management
* **Session Creation and Management:** Create, retrieve, update settings for, and delete conversation sessions.
    * Per-session settings for `alias`, `system_prompt` (system message), `memory_mode` (Enum: `AUTO`, `EXPLICIT`, `DISABLED`) are possible (WIP for mode logic).
    * UUID-based session IDs.
* **Session Management Endpoints:**
    * `POST /sessions/`: Create a new session.
    * `GET /sessions/`: Retrieve a list of sessions.
    * `GET /sessions/{session_id}`: Retrieve details of a specific session.
    * `PUT /sessions/{session_id}`: Modify session information.
    * `DELETE /sessions/{session_id}`: Delete a session.

### 💬 Message and Conversation Management
* **Persistent Message Storage:** Stores detailed message information in the database, including role (`user`, `assistant`, `system`), content (`content`), file references (`file_reference`, `original_filename`, `mime_type`).
* **Conversation Flow Tracking:** Stores `parent_message_id` to maintain the order and parent-child relationships between messages.
* **LLM Metadata Storage:** Records the model used (`model_used`), response time (`response_time_ms`), and tokens used (`prompt_tokens`, `completion_tokens`) for assistant-generated messages.
* **In-Session Message Management Endpoints:**
    * `POST /sessions/{session_id}/messages/`: Add a new message to a session (mainly user messages).
    * `GET /sessions/{session_id}/messages/`: Retrieve a list of messages within a session.
* **Individual Message Retrieval APIs:**
    * `GET /messages/{message_id}`: Retrieve detailed information for a specific message.
    * `GET /messages/{message_id}/history`: Retrieve the history chain from a specific message up to its root parent.

### 🧠 LLM Interaction API
* **Text-based Completion (`POST /llm/text-completion/`):**
    * Supports single text input.
    * Differentiates behavior based on presence of `session_id` (Stateless/Stateful):
        * Stateful: Uses conversation history, system prompt, and memory content with LLM, and saves request/response messages to DB.
        * Stateless: One-off call and response without DB storage.
* **Message List-based Completion (`POST /llm/chat-completions/`):**
    * Accepts a full list of messages (e.g., `List[Message]`) as input for LLM call.
    * Stateless operation (no DB storage, `session_id` not required).
* **Image Input-based Completion (`POST /llm/image-completion/`):**
    * Accepts image files (`multipart/form-data`) and optional text prompts.
    * Currently supports Google Gemini Vision and OpenAI Vision models.
    * Differentiates behavior based on presence of `session_id` (Stateful/Stateless) (saves image reference information in user message if stateful).
* **Text Embedding Generation (`POST /llm/embeddings/`):**
    * Accepts single text or a list of texts to generate embedding vectors.
    * Currently supports OpenAI embedding models.

### 🔧 External Tools (MCP) Integration Readiness
* **Current Method:**
    * A shared MCP Client is instantiated globally in the server.
* **Per-User MCP Server URL Registration/Management API (WIP):**
    * `POST /mcp-connections/`: Register new MCP server URL information.
    * `GET /mcp-connections/`: Retrieve a list of registered MCP server URLs.
    * `GET /mcp-connections/{connection_id}`: Retrieve details of a specific MCP server URL.
    * `PUT /mcp-connections/{connection_id}`: Modify information.
    * `DELETE /mcp-connections/{connection_id}`: Delete information.
    * (Actual MCP tool calling logic is planned to be designed as 'process one function/MCP calling per turn without mini-turns and then return the result'.)

### 💾 Memory Management (User-Driven)
* **Per-Session Memory Mode (WIP for full logic):** Settings for `AUTO`, `EXPLICIT`, `DISABLED` are possible.
* **Memory Entry DB Storage:** Structured memory (type, scope, content, related message IDs, keywords, importance) stored in the `MemoryEntry` table.
* **Manual Memory Management API:**
    * `POST /sessions/{session_id}/memory`: Create a new memory entry.
    * `GET /memory/`: Retrieve a list of memory entries (supports filtering by `session_id`, `scope`, `memory_type`).
    * `GET /memory/{memory_id}`: Retrieve details of a specific memory entry.
    * `PATCH /memory/{memory_id}`: Modify a specific memory entry.
    * `DELETE /memory/{memory_id}`: Delete a specific memory entry.
* **LLM Prompt Injection:** When calling LLMs, if the session's memory mode is not `DISABLED`, relevant memories (session scope + shared scope) are retrieved and automatically injected into the prompt.

### 💰 Usage and Cost Management
* **Estimated Cost Calculation API (`GET /usage/cost-estimation/`):**
    * Calculates estimated usage costs based on `model_used`, `prompt_tokens`, and `completion_tokens` of assistant messages stored in the database.
    * Supports filtering by date range, session ID, provider, and model.

## Setup and Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mori-mmmm/Rini-API-server](https://github.com/mori-mmmm/Rini-API-server)
    cd Rini-API-server
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install required libraries:**
    ```bash
    pip install -r requirements.txt 
    ```
4.  **Set up `.env` file:**
    * Create a `.env` file in the project root and fill in the values based on the following example:
        ```env
        RINI_API_MASTER_KEY= # 32-byte random key for AES encryption (base64 encoded)
        JWT_SECRET_KEY= # Strong secret key for JWT signing
        JWT_ALGORITHM=HS256
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
        ```
5.  **Database Migration:**
    ```bash
    alembic upgrade head
    ```
6.  **Run the server:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
7.  **API Documentation:** Access via browser at `http://localhost:8000/docs` or `http://localhost:8000/redoc`.

## Future Plans (Remaining Work & Improvements)
* **Memory Functionality Enhancement:** Implement `AUTO` mode logic (automatic summarization/extraction), and additional `EXPLICIT` mode APIs (e.g., summarize based on specific conversation).
* **LLM Feature Expansion:**
    * Support for image input and embedding for other providers (Gemini, Anthropic).
    * Functionality for handling general files (foundation for RAG).
* **Error Handling Improvement:** More granular error handling for LLM quota issues, etc., and standardization of error response structure.
* **Detailed Logging Application:** Enhance logging across the application for key events and errors.
* Conversation Branching feature
* Session History Retrieval (Tree format)
* RAG (Retrieval Augmented Generation) - File upload based
* Agent functionality

## License
MIT License

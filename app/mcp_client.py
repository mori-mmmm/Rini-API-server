import asyncio
import json
import os
from typing import Optional, List, Dict
from contextlib import AsyncExitStack



from mcp import ClientSession, Tool as MCPTool

from mcp.client.sse import sse_client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

MCP_SERVER_URL_LIST = [
    "http://localhost:65000/sse",
]

class MCPClient:
    def __init__(self):
        """MCP 다중 클라이언트 (SSE) 초기화"""
        self.sessions: Dict[str, ClientSession] = {}
        self.available_tools: List[Dict] = []
        self.tool_to_session_map: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()






        loop = asyncio.get_event_loop()
        loop.create_task(self.connect_to_servers())

    async def connect_to_servers(self):
        """SSE를 사용하여 여러 MCP 서버에 연결하고 세션을 초기화합니다."""
        successful_connections = 0

        for url in MCP_SERVER_URL_LIST:
            print(url)
            try:

                streams_context = sse_client(url=url)
                streams = await self.exit_stack.enter_async_context(streams_context)

                session_context = ClientSession(*streams)
                session: ClientSession = await self.exit_stack.enter_async_context(
                    session_context
                )


                await session.initialize()
                self.sessions[url] = session
                successful_connections += 1


                response = await session.list_tools()
                session_tools = response.tools

                for tool in session_tools:
                    print(f"desc for {tool.name}: {tool.description} {tool.inputSchema}")
                    tool_definition = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                    print(tool.name, tool.description)



                    if tool.name not in self.tool_to_session_map:
                         self.available_tools.append(tool_definition)
                    else:
                         print(f"  [경고] 도구 '{tool.name}'이(가) 이미 다른 서버에 존재합니다. 이 서버의 정의({url})를 사용합니다.")

                         existing_index = next((i for i, t in enumerate(self.available_tools) if t['function']['name'] == tool.name), -1)
                         if existing_index != -1:
                             self.available_tools[existing_index] = tool_definition



                    self.tool_to_session_map[tool.name] = session

            except Exception as e:
                print(f"  [실패] 서버(SSE) 연결 중 오류 발생 ({url}): {type(e).__name__}: {e}")

        print(f"\n--- 연결 요약 (SSE) ---")
        
        if self.sessions:
            print(f"성공적으로 연결된 서버: {list(self.sessions.keys())}")
            print(f"사용 가능한 총 고유 도구: {[tool['function']['name'] for tool in self.available_tools]}")

            all_tool_names = []
            for session in self.sessions.values():
                try:
                    tools_response = await session.list_tools()
                    all_tool_names.extend([tool.name for tool in tools_response.tools])
                except Exception as e:
                    print(f" [경고] 연결 요약 중 도구 목록 조회 실패 ({getattr(session, 'url', 'Unknown URL')}): {e}")

            if len(all_tool_names) != len(set(all_tool_names)):
                 print("[경고] 여러 서버에서 동일한 이름의 도구가 발견되었습니다. 마지막으로 연결된 서버의 도구가 사용됩니다.")
        else:
            print("성공적으로 연결된 MCP 서버(SSE)가 없습니다.")

    async def cleanup(self):
        """AsyncExitStack에 의해 관리되는 모든 리소스를 정리합니다."""
        print("\n리소스 정리 중...")
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.available_tools.clear()
        self.tool_to_session_map.clear()
        print("리소스 정리 완료.")

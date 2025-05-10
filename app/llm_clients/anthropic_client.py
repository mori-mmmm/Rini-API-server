



import os
import logging
import asyncio
from typing import List, Dict, Optional, Any

from anthropic import AsyncAnthropic, APIError, RateLimitError, AuthenticationError
from .util import get_tool_prompt, get_json

MAX_TOOL_RETRIES = 3

logger = logging.getLogger("app.llm_clients.anthropic")


def get_response_text(response):
    if response.content and isinstance(response.content, list) and len(response.content) > 0:
        first_block = response.content[0]
        if hasattr(first_block, 'text'):
            response_text = first_block.text
            return response_text
    return ""

async def handle_tool_call(client, mcp_client, model, tool_choice, effective_system_prompt, history_messages, max_tokens, **kwargs):
    prompt = get_tool_prompt(mcp_client.available_tools, tool_choice)
    current_message = {
        "role": "user",
        "content": prompt
    }
    history_messages.append(current_message)

    response = await client.messages.create(
        model=model,
        system=effective_system_prompt if effective_system_prompt else None,
        messages=history_messages,
        max_tokens=max_tokens,
        **kwargs
    )
    

    response_text = get_response_text(response)
    completion = get_json(response_text)
    if "tool_calls" in completion and completion["tool_calls"]:
        results = []
        tool_calls = completion["tool_calls"]

        for tool_call in tool_calls:
            tool_args = tool_call["args"]
            tool_name = tool_call["name"]

            target_session = mcp_client.tool_to_session_map.get(tool_name)
            if not target_session:
                tool_result_content = f"[오류: 도구 '{tool_name}'을(를) 찾을 수 없음]"
                results.append({
                    "tool_name": tool_name,
                    "tool_result": tool_result_content
                })
                continue

            last_exception = None
            for attempt in range(MAX_TOOL_RETRIES):
                try:
                    print(f"      - target_session.call_tool('{tool_name}') 대기 중...")
                    result = await target_session.call_tool(tool_name, tool_args)
                    print(f"      - '{tool_name}'에 대한 await 완료.")
                    print(f"      - 결과 객체 타입: {type(result)}")

                    if result is None:
                        raise ValueError(f"도구 '{tool_name}'이 예기치 않게 None을 반환했습니다.")

                    print(f"      - '{tool_name}'의 result.content 접근 중...")
                    if hasattr(result, "content"):
                        tool_result_content = result.content
                        print(f"      - '{tool_name}'의 result.content 접근 성공. 타입: {type(tool_result_content)}")
                    else:
                        print(f"      - [오류] '{tool_name}'의 결과 객체에 'content' 속성이 없습니다.")
                        raise AttributeError(f"도구 '{tool_name}'의 결과 객체에 'content' 속성이 없습니다. 결과: {result}")

                    content_str = str(tool_result_content)
                    print(f"    결과 내용 (처음 1000자): {content_str[:1000]}{'...' if len(content_str) > 1000 else ''}")
                    last_exception = None
                    break

                except Exception as e:
                    print(f"    [오류] 도구 호출 '{tool_name}' 실패 (시도 {attempt + 1}/{MAX_TOOL_RETRIES}): {type(e).__name__}: {str(e)}")
                    last_exception = e
                    if attempt == MAX_TOOL_RETRIES - 1:
                        print(f"    도구 '{tool_name}'이(가) {MAX_TOOL_RETRIES}번의 시도 후 실패했습니다.")
                        tool_result_content = f"[오류: 도구 {tool_name} 실행 중 {MAX_TOOL_RETRIES}번 시도 후 오류 발생: {type(last_exception).__name__}: {str(last_exception)}]"
                    else:
                        await asyncio.sleep(1)


            print(f"  - 도구 '{tool_name}' 결과 추가 중... 성공: {last_exception is None}).")
            results.append({
                "tool_name": tool_name,
                "tool_result": tool_result_content
            })
            print(f"  - 도구 '{tool_name}' 처리 완료.")
    
    
        response_text = get_response_text(response)
        current_messages = [
            {
                "role": "assistant",
                "content": response_text
            },
            {
                "role": "user",
                "content":  f"{results}\n위 tool_call 실행 결과를 가지고 원래 질문에 답해줘"
            }
        ]
        history_messages += current_messages

        response = await client.messages.create(
            model=model,
            system=effective_system_prompt if effective_system_prompt else None,
            messages=history_messages,
            max_tokens=max_tokens,
            **kwargs
        )
    return response

async def generate_text_from_messages_async(
    api_key: str,
    messages: List[Dict],
    model: str = "claude-3-opus-20240229",
    system_prompt: Optional[str] = None,
    max_tokens: int = 16384,
    tool_choice: Optional[bool] = None,
    mcp_client: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """Anthropic Claude API를 비동기적으로 호출하여 메시지 목록 기반 텍스트 응답 생성"""
    logger.info(f"Anthropic API 호출 시작: model='{model}', messages_count={len(messages)}, max_tokens={max_tokens}, kwargs={kwargs}")
    if system_prompt:
        logger.info("  시스템 프롬프트 제공됨.")
    else:
        logger.info("  시스템 프롬프트 제공되지 않음.")

    try:
        logger.debug("Anthropic Async 클라이언트 초기화 중...")
        client = AsyncAnthropic(api_key=api_key)
        logger.debug("Anthropic Async 클라이언트 초기화 완료.")


        history_messages = []
        effective_system_prompt = system_prompt
        logger.debug("메시지 목록 처리 중...")
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            logger.debug(f"  메시지 {i+1}: role='{role}', content_length={len(content) if content else 0}")
            if role in ["user", "assistant"] and content:
                history_messages.append({"role": role, "content": content})
            elif role == "system" and not effective_system_prompt and content:
                 logger.info("  messages 리스트 내 system 메시지를 시스템 프롬프트로 사용합니다.")
                 effective_system_prompt = content
        logger.debug(f"처리된 히스토리 메시지 수: {len(history_messages)}")
        if effective_system_prompt:
            logger.debug(f"  최종 시스템 프롬프트 길이: {len(effective_system_prompt)}")


        logger.debug(f"Anthropic messages.create API 호출 (model='{model}', max_tokens={max_tokens})")

        if tool_choice in ["auto", "required"]:
            response = await handle_tool_call(client, mcp_client, model, tool_choice, effective_system_prompt, history_messages, max_tokens, **kwargs)
        else:
            response = await client.messages.create(
                model=model,
                system=effective_system_prompt if effective_system_prompt else None,
                messages=history_messages,
                max_tokens=max_tokens,
                **kwargs
            )
        logger.debug("Anthropic API 응답 수신 완료.")


        response_text = ""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, 'text'):
                response_text = first_block.text

        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'input_tokens', None)
            completion_tokens = getattr(response.usage, 'output_tokens', None)
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            logger.info(f"  Anthropic 토큰 사용량 - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
        else:
            logger.warning("Anthropic API 응답에서 토큰 사용량 정보를 찾을 수 없습니다.")

        result = {
            "text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        logger.info(f"Anthropic API 호출 성공. 응답 텍스트 길이: {len(response_text)}")
        logger.debug(f"  반환 결과: {result}")
        return result


    except AuthenticationError as e:
        logger.error(f"Anthropic API 인증 오류: {e}", exc_info=True)
        raise ValueError("Anthropic API 키가 유효하지 않습니다.") from e
    except RateLimitError as e:
        logger.error(f"Anthropic API 쿼터 초과 오류: {e}", exc_info=True)
        raise ConnectionAbortedError("Anthropic API 쿼터를 초과했습니다.") from e
    except APIError as e:
        logger.error(f"Anthropic API 오류 발생: {e}", exc_info=True)
        error_message = f"Anthropic API 오류 (status: {e.status_code if hasattr(e, 'status_code') else 'N/A'}): {e}"
        raise ConnectionError(error_message) from e
    except Exception as e:
        logger.exception(f"Anthropic 클라이언트 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"Anthropic 처리 오류: {e}") from e

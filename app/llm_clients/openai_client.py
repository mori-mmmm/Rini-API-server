

import os
import logging
from typing import List, Dict, Optional, Any
import base64
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError

from .util import get_tool_prompt, get_json

MAX_TOOL_RETRIES = 3

logger = logging.getLogger("app.llm_clients.openai")

def get_response_text(completion):
    if completion.choices:
        response_message = completion.choices[0].message
        if response_message and response_message.content:
            response_text = response_message.content
            return response_text
    return ""

async def handle_tool_call(client, mcp_client, model, tool_choice, messages, **kwargs):
    prompt = get_tool_prompt(mcp_client.available_tools, tool_choice)
    current_message = {
        "role": "user",
        "content": prompt
    }
    messages.append(current_message)

    orig_completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    
    response_text = get_response_text(orig_completion)
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
    
    
        response_text = get_response_text(orig_completion)
        current_messages = [
            {
                "role": "assistant",
                "content": response_text
            },
            {
                "role": "user",
                "content": str(results)
            }
        ]
        messages += current_messages

        orig_completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    return orig_completion


async def generate_text_from_messages_async(
    api_key: str,
    messages: List[Dict],
    model: str = "gpt-4o",
    tool_choice: Optional[bool] = None,
    mcp_client: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """OpenAI Chat Completions API를 비동기적으로 호출 후 텍스트와 토큰 사용량 반환"""
    logger.info(f"OpenAI generate_text_from_messages_async 호출: model='{model}', messages_count={len(messages)}, kwargs={kwargs}")
    try:
        logger.debug("OpenAI Async 클라이언트 초기화 중...")
        client = AsyncOpenAI(api_key=api_key)
        logger.debug("OpenAI Async 클라이언트 초기화 완료.")

        logger.debug(f"OpenAI client.chat.completions.create API 호출 (model='{model}')")

        if tool_choice in ["auto", "required"]:
            completion = await handle_tool_call(client, mcp_client, model, tool_choice, messages, **kwargs)
        else:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )


        response_text = ""
        if completion.choices:
            response_message = completion.choices[0].message
            if response_message and response_message.content:
                response_text = response_message.content
            else:
                finish_reason = completion.choices[0].finish_reason
                logger.warning(f"OpenAI response finished with reason: {finish_reason}, but no content found.")
        else:
            logger.warning("OpenAI API 응답에 선택지(choices)가 없습니다.")

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if hasattr(completion, 'usage') and completion.usage:
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', None)
            completion_tokens = getattr(completion.usage, 'completion_tokens', None)
            total_tokens = getattr(completion.usage, 'total_tokens', None)
            logger.info(f"  OpenAI Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
        else:
            logger.warning("OpenAI API 응답에서 토큰 사용량(usage) 정보를 찾을 수 없습니다.")

        return_value = {
            "text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        logger.info(f"generate_text_from_messages_async 반환: text_len={len(response_text)}, tokens={total_tokens}")
        return return_value


    except AuthenticationError as e:
        logger.error(f"OpenAI API 인증 오류: {e}", exc_info=True)
        raise ValueError("OpenAI API 키가 유효하지 않습니다.") from e
    except RateLimitError as e:
        logger.error(f"OpenAI API 쿼터 초과 오류: {e}", exc_info=True)
        raise ConnectionAbortedError("OpenAI API 쿼터를 초과했습니다.") from e
    except APIError as e:
        logger.error(f"OpenAI API 오류 발생: {e}", exc_info=True)
        raise ConnectionError(f"OpenAI API 오류: {e}") from e
    except Exception as e:
        logger.exception(f"OpenAI 클라이언트 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"OpenAI 처리 오류: {e}") from e



async def generate_text_from_text_async(
    api_key: str,
    prompt: str,
    model: str = "gpt-4o",
    **kwargs
) -> Dict[str, Any]:
    """단일 텍스트 입력을 받아 비동기 OpenAI 호출 후 결과 딕셔너리 반환"""
    logger.info(f"OpenAI generate_text_from_text_async 호출: model='{model}', prompt_length={len(prompt)}, kwargs={kwargs}")
    messages = [{"role": "user", "content": prompt}]

    return await generate_text_from_messages_async(
        api_key=api_key,
        messages=messages,
        model=model,
        **kwargs
    )


async def get_embedding_async(
    api_key: str,
    text: str,
    model: str = "text-embedding-3-large"
) -> Dict[str, Any]:
    """텍스트 하나에 대한 임베딩 벡터를 비동기적으로 생성합니다."""
    logger.info(f"OpenAI get_embedding_async 호출: model='{model}', text_length={len(text)}")
    try:
        logger.debug("OpenAI Async 클라이언트 초기화 중 (임베딩)...")
        client = AsyncOpenAI(api_key=api_key)
        logger.debug("OpenAI Async 클라이언트 초기화 완료 (임베딩).")

        logger.debug(f"OpenAI client.embeddings.create API 호출 (model='{model}')")
        response = await client.embeddings.create(input=[text], model=model)
        logger.debug("OpenAI Embeddings API 응답 수신 완료.")


        if not response.data:
            logger.error("OpenAI Embeddings API 응답에 데이터가 없습니다.")
            raise ValueError("OpenAI Embeddings API 응답에 데이터가 없습니다.")
        
        embedding_object = response.data[0]
        usage_data = response.usage
        prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
        total_tokens = getattr(usage_data, 'total_tokens', None)
        logger.info(f"  OpenAI Embedding Tokens - Prompt: {prompt_tokens}, Total: {total_tokens}")

        return_value = {
            "data": [
                {
                    "embedding": embedding_object.embedding,
                    "index": embedding_object.index
                }
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
        logger.info(f"get_embedding_async 반환: model='{response.model}', tokens={total_tokens}")
        return return_value


    except AuthenticationError as e:
        logger.error(f"OpenAI API 인증 오류 (임베딩): {e}", exc_info=True)
        raise ValueError("OpenAI API 키가 유효하지 않습니다.") from e
    except RateLimitError as e:
        logger.error(f"OpenAI API 쿼터 초과 오류 (임베딩): {e}", exc_info=True)
        raise ConnectionAbortedError("OpenAI API 쿼터를 초과했습니다.") from e
    except APIError as e:
        logger.error(f"OpenAI API 오류 발생 (임베딩): {e}", exc_info=True)
        raise ConnectionError(f"OpenAI API 오류: {e}") from e
    except Exception as e:
        logger.exception(f"OpenAI 임베딩 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"OpenAI 임베딩 처리 오류: {e}") from e



async def get_embedding_batch_async(
    api_key: str,
    texts: List[str],
    model: str = "text-embedding-3-large"
) -> Dict[str, Any]:
    """텍스트 목록에 대한 임베딩 벡터 목록을 비동기적으로 생성합니다."""
    logger.info(f"OpenAI get_embedding_batch_async 호출: model='{model}', texts_count={len(texts)}")
    if not texts:
        logger.warning("입력 텍스트 목록이 비어 있어 빈 결과를 반환합니다.")
        return {"data": [], "model": model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}
    try:
        logger.debug("OpenAI Async 클라이언트 초기화 중 (배치 임베딩)...")
        client = AsyncOpenAI(api_key=api_key)
        logger.debug("OpenAI Async 클라이언트 초기화 완료 (배치 임베딩).")

        logger.debug(f"OpenAI client.embeddings.create API 호출 (배치, model='{model}')")
        response = await client.embeddings.create(input=texts, model=model)
        logger.debug("OpenAI Embeddings API 응답 수신 완료 (배치).")


        if not response.data or len(response.data) != len(texts):
            logger.error(f"입력 텍스트 수({len(texts)})와 응답 임베딩 수({len(response.data) if response.data else 0})가 일치하지 않습니다 (async).")
            raise ValueError("OpenAI Embeddings API 응답 데이터 수가 입력 텍스트 수와 일치하지 않습니다 (async).")

        embedding_data_list = [
            {"embedding": item.embedding, "index": item.index}
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        logger.debug(f"생성된 임베딩 데이터 리스트: {len(embedding_data_list)}개 항목")

        usage_data = response.usage
        prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
        total_tokens = getattr(usage_data, 'total_tokens', None)
        logger.info(f"  OpenAI Embedding Batch Tokens - Prompt: {prompt_tokens}, Total: {total_tokens}")

        return_value = {
            "data": embedding_data_list,
            "model": response.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }
        logger.info(f"get_embedding_batch_async 반환: model='{response.model}', items={len(embedding_data_list)}, tokens={total_tokens}")
        return return_value


    except AuthenticationError as e:
        logger.error(f"OpenAI API 인증 오류 (배치 임베딩): {e}", exc_info=True)
        raise ValueError("OpenAI API 키가 유효하지 않습니다.") from e
    except RateLimitError as e:
        logger.error(f"OpenAI API 쿼터 초과 오류 (배치 임베딩): {e}", exc_info=True)
        raise ConnectionAbortedError("OpenAI API 쿼터를 초과했습니다.") from e
    except APIError as e:
        logger.error(f"OpenAI API 오류 발생 (배치 임베딩): {e}", exc_info=True)
        raise ConnectionError(f"OpenAI API 오류: {e}") from e
    except Exception as e:
        logger.exception(f"OpenAI 배치 임베딩 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"OpenAI 배치 임베딩 처리 오류: {e}") from e


async def generate_text_from_image_and_text_async(
    api_key: str,
    image_bytes: bytes,
    mime_type: str,
    prompt: Optional[str],
    model: str = "gpt-4o",
    max_tokens: Optional[int] = 3000,
    **kwargs
) -> Dict[str, Any]:
    """OpenAI Vision 모델을 비동기적으로 호출하여 이미지와 텍스트 기반 응답 생성"""
    logger.info(f"OpenAI generate_text_from_image_and_text_async 호출: model='{model}', prompt_len={len(prompt or '')}")
    if not prompt:
        prompt = "Describe this image."

    try:
        logger.debug("OpenAI Async 클라이언트 초기화 중 (Vision)...")
        client = AsyncOpenAI(api_key=api_key)
        logger.debug("OpenAI Async 클라이언트 초기화 완료 (Vision).")


        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url_data = f"data:{mime_type};base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url_data}}
                ]
            }
        ]

        logger.debug(f"OpenAI client.chat.completions.create API 호출 (Vision async, model='{model}')")
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
        logger.debug("OpenAI Vision API 응답 수신 완료.")


        response_text = ""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if completion.choices:
            response_message = completion.choices[0].message
            if response_message and response_message.content:
                response_text = response_message.content
            else:
                 finish_reason = completion.choices[0].finish_reason
                 logger.warning(f"OpenAI Vision response finished with reason: {finish_reason}, but no content found.")
        else:
            logger.warning("OpenAI Vision API 응답에 선택지(choices)가 없습니다.")

        if hasattr(completion, 'usage') and completion.usage:
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', None)
            completion_tokens = getattr(completion.usage, 'completion_tokens', None)
            total_tokens = getattr(completion.usage, 'total_tokens', None)
            logger.info(f"  OpenAI Vision Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        return_value = {
            "text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        logger.info(f"generate_text_from_image_and_text_async 반환: text_len={len(response_text)}, tokens={total_tokens}")
        return return_value


    except AuthenticationError as e:
        logger.error(f"OpenAI Vision API 인증 오류: {e}")
        raise ValueError("OpenAI API 키가 유효하지 않습니다.") from e
    except RateLimitError as e:
        logger.error(f"OpenAI Vision API 쿼터 초과 오류: {e}")
        raise ConnectionAbortedError("OpenAI API 쿼터를 초과했습니다.") from e
    except APIError as e:
        logger.error(f"OpenAI Vision API 오류 발생: {e}")
        raise ConnectionError(f"OpenAI API 오류: {e}") from e
    except Exception as e:
        logger.exception(f"OpenAI Vision 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"OpenAI Vision 처리 오류: {e}") from e



import os
import base64
import mimetypes
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from google import genai

from google.generativeai import types
from google.generativeai import generative_models
from google.api_core import exceptions as google_exceptions

from .util import get_tool_prompt, get_json

MAX_TOOL_RETRIES = 3

load_dotenv()
logger = logging.getLogger("app.llm_clients.google")



def _prepare_gemini_contents(messages: List[Dict]) -> List[Dict[str, Any]]:
    """FastAPI 메시지 형식을 Gemini API 콘텐츠 형식으로 변환 (동기 함수 유지)"""

    contents = []
    current_content = None
    for message in messages:
        role = message.get("role", "user")
        content_text = message.get("content")
        if not content_text:
            continue

        gemini_role = "user"
        if role == "assistant":
            gemini_role = "model"
        elif role == "system":
            gemini_role = "user"

        text_part = {"text": content_text}

        if current_content and current_content["role"] == gemini_role:
            current_content["parts"].append(text_part)
        else:
            current_content = {
                "role": gemini_role,
                "parts": [text_part]
            }
            contents.append(current_content)
    return contents


def get_response_text(response):
    if hasattr(response, 'text') and response.text:
        response_text = response.text
    elif response.candidates:
        first_candidate = response.candidates[0]
        if first_candidate.content and first_candidate.content.parts and first_candidate.content.parts[0].text:
            response_text = first_candidate.content.parts[0].text
    return response_text

async def handle_tool_call(client, mcp_client, model, gen_config_dict, tool_choice, contents):
    prompt = get_tool_prompt(mcp_client.available_tools, tool_choice)
    current_content = {
        "role": "user",
        "parts": [{"text": prompt}]
    }
    contents.append(current_content)

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=gen_config_dict if gen_config_dict else None,

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
        current_contents = [
            {
                "role": "model",
                "parts": [{"text": response_text}]
            },
            {
                "role": "user",
                "parts": [{"text": str(results)}]
            }
        ]
        contents += current_contents

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config_dict if gen_config_dict else None

        )
    return response

async def generate_text_from_messages_async(
    api_key: str,
    messages: List[Dict],
    model: str,
    thinking_budget: Optional[int] = None,
    generation_config_override: Optional[Dict[str, Any]] = None,
    tool_choice: Optional[bool] = None,
    mcp_client: Any = None,
) -> Dict[str, Any]:
    """Gemini API를 비동기적으로 호출하여 메시지 목록 기반 텍스트 응답 생성."""
    logger.info(f"Gemini generate_text_from_messages_async 호출: model='{model}', messages_count={len(messages)}, thinking_budget={thinking_budget}")
    if generation_config_override:
        logger.info(f"  Generation config override: {generation_config_override}")

    try:





        client = genai.Client(api_key=api_key)
        logger.debug("Gemini API key configured.")

        contents = _prepare_gemini_contents(messages)
        if not contents:
            logger.warning("처리할 유효한 메시지가 없어 ValueError 발생시킵니다.")
            raise ValueError("처리할 유효한 메시지가 없습니다.")
        logger.debug(f"준비된 Gemini contents: {contents}")

        gen_config_dict: Dict[str, Any] = generation_config_override if generation_config_override else {}
        logger.debug(f"사용될 Generation config: {gen_config_dict}")

        if thinking_budget is not None and thinking_budget > 0:
            gen_config_dict['thinking_config'] = {"thinking_budget": thinking_budget}
            logger.info(f"  (Thinking budget {thinking_budget} applied to generation config)")


        if tool_choice in ["auto", "required"]:
            response = await handle_tool_call(client, mcp_client, model, gen_config_dict, tool_choice, contents)
        else:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config_dict if gen_config_dict else None

            )
        logger.debug("Gemini API 응답 수신 완료.")


        response_text = ""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(response.usage_metadata, 'total_token_count', None)
                logger.info(f"  Gemini Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            else:
                logger.warning("Gemini API 응답에서 토큰 사용량 정보를 찾을 수 없습니다.")

            if hasattr(response, 'text') and response.text:
                response_text = response.text
            elif response.candidates:
                first_candidate = response.candidates[0]
                if first_candidate.content and first_candidate.content.parts and first_candidate.content.parts[0].text:
                    response_text = first_candidate.content.parts[0].text
                else:
                    finish_reason = getattr(first_candidate, 'finish_reason', 'Unknown')
                    logger.warning(f"Gemini response finished with reason: {finish_reason}, but no text content found.")
            else:
                 if hasattr(response, 'prompt_feedback'):
                     logger.warning(f"Gemini prompt feedback (no candidates): {response.prompt_feedback}")

        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"Error parsing Gemini response structure: {e}", exc_info=True)
             logger.debug(f"Full Response Object (on parsing error): {response}")
             raise ValueError("Gemini API 응답 구조를 파싱하는 데 실패했습니다.") from e

        return_value = {
            "text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        logger.info(f"generate_text_from_messages_async 반환: text_len={len(response_text)}, tokens={total_tokens}")
        return return_value


    except google_exceptions.PermissionDenied as e:
        logger.error(f"Gemini API 권한 오류: {e.message}", exc_info=True)
        raise ValueError(f"Gemini API 키가 유효하지 않거나 권한이 없습니다: {e.message}") from e
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Gemini API 쿼터 초과 오류: {e.message}", exc_info=True)
        raise ConnectionAbortedError(f"Gemini API 쿼터를 초과했습니다: {e.message}") from e
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API 오류 발생: {e.message}", exc_info=True)
        raise ConnectionError(f"Gemini API 오류: {e.message}") from e
    except Exception as e:
        logger.exception(f"Gemini 클라이언트 처리 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"Gemini 처리 오류: {e}") from e


async def generate_text_from_text_async(
    api_key: str,
    text: str,
    model: str,
    thinking_budget: Optional[int] = None,
    generation_config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """단일 텍스트 입력을 받아 비동기 Gemini API 호출"""
    logger.info(f"generate_text_from_text_async 호출: model='{model}', text_length={len(text)}")
    messages = [{"role": "user", "content": text}]

    return await generate_text_from_messages_async(
        api_key=api_key,
        messages=messages,
        model=model,
        thinking_budget=thinking_budget,
        generation_config_override=generation_config_override
    )



def _read_image_bytes(image_path: str) -> bytes:
    """이미지 파일 경로에서 바이트 데이터를 읽음 (동기)"""

    if not os.path.exists(image_path):
        logger.error(f"_read_image_bytes: 이미지 파일을 찾을 수 없습니다: '{image_path}'")
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: '{image_path}'")
    try:
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            logger.debug(f"_read_image_bytes: '{image_path}' 읽기 성공 ({len(data)} bytes)")
            return data
    except Exception as e:
        logger.exception(f"_read_image_bytes: 이미지 파일 ('{image_path}') 읽기 오류")
        raise IOError(f"이미지 파일 읽기 실패: {image_path}") from e

def _read_image_bytes_and_detect_mime(image_path: str) -> Tuple[bytes, str]:
    """이미지 파일 경로에서 바이트 데이터와 MIME 타입을 읽고 감지합니다 (동기)."""

    logger.debug(f"_read_image_bytes_and_detect_mime 호출: '{image_path}'")
    if not os.path.exists(image_path):
        logger.error(f"이미지 파일을 찾을 수 없습니다: '{image_path}'")
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: '{image_path}'")

    mime_type, _ = mimetypes.guess_type(image_path)
    logger.debug(f"  '{image_path}' 추측된 MIME 타입: {mime_type}")
    supported_image_types = ['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif']

    if not mime_type or mime_type not in supported_image_types:
        logger.warning(f"  '{image_path}'의 MIME 타입({mime_type})이 지원되지 않거나 알 수 없습니다.")
        raise ValueError(
            f"이미지 파일의 MIME 타입을 감지할 수 없거나 지원되지 않는 형식입니다 (감지된 타입: {mime_type}). "
            f"경로: '{image_path}'. 지원 형식: {', '.join(supported_image_types)}"
        )

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        logger.info(f"  '{image_path}' 읽기 및 MIME 타입 감지 성공: {mime_type}, {len(image_bytes)} bytes")
        return image_bytes, mime_type
    except Exception as e:
        logger.exception(f"이미지 파일 ('{image_path}') 읽기 중 오류")
        raise IOError(f"이미지 파일 읽기 실패: {image_path}") from e


async def generate_text_from_image_and_text_async(
    api_key: str,
    image_path: str,
    prompt: str,
    generation_config_override: Optional[Dict[str, Any]] = None,
    tool_choice: Optional[bool] = None,
    mcp_client: Any = None,
) -> Dict[str, Any]:
    """이미지와 텍스트 프롬프트를 받아 비동기적으로 Gemini 멀티모달 모델 호출"""
    fixed_model_name = "gemini-2.0-flash"
    logger.info(f"generate_text_from_image_and_text_async 호출: model='{fixed_model_name}', image_path='{image_path}', prompt_length={len(prompt)}")
    if generation_config_override:
        logger.info(f"  Generation config override: {generation_config_override}")

    try:

        client = genai.Client(api_key=api_key)
        logger.debug("Gemini API key configured (multimodal).")


        image_bytes, detected_mime_type = _read_image_bytes_and_detect_mime(image_path)
        logger.info(f"  이미지 로드: '{os.path.basename(image_path)}', MIME: {detected_mime_type}")

        image_part = types.Part.from_bytes(mime_type=detected_mime_type, data=image_bytes)
        text_part = types.Part.from_text(text=prompt)
        logger.debug("  이미지 및 텍스트 파트 생성 완료.")

        contents = [types.Content(role="user", parts=[image_part, text_part])]
        logger.debug(f"  멀티모달 contents: {contents}")

        gen_config_dict = generation_config_override if generation_config_override else {}
        final_generation_config = types.GenerationConfig(**gen_config_dict) if gen_config_dict else None
        logger.debug(f"  멀티모달 Generation config: {final_generation_config}")


        if tool_choice in ["auto", "required"]:
            response = await handle_tool_call(client, mcp_client, fixed_model_name, final_generation_config, tool_choice, contents)
        else:
            response = await client.aio.models.generate_content(
                model=fixed_model_name,
                contents=contents,
                config=final_generation_config

            )
        logger.debug("Gemini API 응답 수신 완료 (멀티모달).")


        response_text = ""
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(response.usage_metadata, 'total_token_count', None)
                logger.info(f"  Gemini Tokens (multimodal) - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            else:
                logger.warning("Gemini API 응답(멀티모달)에서 토큰 사용량 정보를 찾을 수 없습니다.")

            if hasattr(response, 'text') and response.text:
                response_text = response.text
            elif response.candidates:
                 first_candidate = response.candidates[0]
                 if first_candidate.content and first_candidate.content.parts and first_candidate.content.parts[0].text:
                    response_text = first_candidate.content.parts[0].text
                 else:
                    finish_reason = getattr(first_candidate, 'finish_reason', 'Unknown')
                    logger.warning(f"Gemini response (multimodal) finished with reason: {finish_reason}, but no text content found.")

        except (AttributeError, IndexError, TypeError) as e:
             logger.error(f"Error parsing Gemini response structure (multimodal): {e}", exc_info=True)
             try: logger.debug(f"Full Response Object (multimodal, on parsing error): {response}")
             except Exception: logger.debug("Could not print the full response object (multimodal).")


        return_value = {
            "text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        logger.info(f"generate_text_from_image_and_text_async 반환: text_len={len(response_text)}, tokens={total_tokens}")
        return return_value


    except FileNotFoundError as e:
        logger.error(f"이미지 파일 접근 불가: {e.filename}", exc_info=True)
        raise
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API 오류 발생 (멀티모달): {e.message}", exc_info=True)
        raise ConnectionError(f"Gemini API 오류 (멀티모달): {e.message}") from e
    except Exception as e:
        logger.exception(f"Gemini 이미지 처리 중 예기치 않은 오류 발생 : {e}")
        raise RuntimeError(f"Gemini 이미지 처리 오류 : {e}") from e

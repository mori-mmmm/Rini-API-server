import json
import re

def get_tool_prompt(tools, tool_choice):
    prompt = f"""
당신이 사용할 수 있는 tool의 목록과 각각의 용도와 인자들은 다음과 같습니다:
{tools}

{"적어도 하나의 tool을 무조건 사용해야 합니다." if tool_choice == "required" else "tool을 사용할지 스스로 결정하세요."}
tool을 사용하는 경우 **코드 블럭에 넣어서** json 포맷으로 바로 생성하세요.
출력 포맷은 다음과 같습니다:

하고 싶은 말이나 설명을 덧붙여도 좋습니다
```json
{{
    tool_calls: [
        {{
            name: (함수 이름),
            args: {{
                (인자 이름1): (인자 값1),
                (인자 이름2): (인자 값2),
                ...
            }}
        }},
        ...
    ]
}}
```
"""
    return prompt

def get_json(text):
    print(text)
    try:
        text = text.split("```json")[1].split("```")[0]
    except:
        return "(아래 내용에서 json 코드블럭을 발견하지 못해서 함수호출은 실행되지 않습니다.)\n" + text
    code_block_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
    match = code_block_pattern.search(text)
    
    if match:
        json_text = match.group(1)
    else:
        json_text = text

    try:
        parsed_json = json.loads(json_text)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
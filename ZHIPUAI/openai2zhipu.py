# ----------------------------------------------------------------
# ZhipuAI API Methods
# ----------------------------------------------------------------
from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key="your ZhipuAI api key"
)  # 如果您使用 智谱AI 的SDK，请使用这个代码

# ----------------------------------------------------------------
# OpenAI API Methods
# ----------------------------------------------------------------
from openai import OpenAI

client = OpenAI(
    api_key="your ZhipuAI api key", base_url="https://open.bigmodel.cn/api/paas/v4"
)


def function_chat(use_stream=False):
    messages = [
        {"role": "user", "content": "What's the Celsius temperature in San Francisco?"},
        # Give Observations
        {
            "role": "assistant",
            "content": None,
            "function_call": None,
            "tool_calls": [
                {
                    "id": "call_1717912616815",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "San Francisco, CA", "format": "celsius"}',
                    },
                    "type": "function",
                }
            ],
        },
        ## Add Observation Result if you need
        # {
        #     "tool_call_id": "call_1717912616815",
        #     "role": "tool",
        #     "name": "get_current_weather",
        #     "content": "23°C",
        # }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
        stream=use_stream,
        max_tokens=256,
        temperature=0.9,
        presence_penalty=1.2,
        top_p=0.1,
        tool_choice="auto",
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk)
        else:
            print(response)
    else:
        print("Error:", response.status_code)


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "system",
            "content": "请在你输出的时候都带上“喵喵喵”三个字，放在开头。",
        },
        {"role": "user", "content": "你是谁"},
    ]
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.4,
        presence_penalty=1.2,
        top_p=0.8,
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk)
        else:
            print(response)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    simple_chat(use_stream=False)
    function_chat(use_stream=False)

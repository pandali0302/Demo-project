# --------------------------------------------------------------
# Import Modules
# --------------------------------------------------------------

import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain_core.tools import tool
from zhipuai import ZhipuAI


# --------------------------------------------------------------
# Load zhipuai API Token From the .env File
# --------------------------------------------------------------

_ = load_dotenv(find_dotenv())

api_key = os.getenv("ZHIPUAI_API_KEY")


# --------------------------------------------------------------
# Ask ChatGLM a Question
# --------------------------------------------------------------


client = ZhipuAI(api_key=api_key)
messages = []
messages.append(
    {"role": "user", "content": "帮我查询从2024年1月20日，从北京出发前往上海的航班"}
)
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=messages,
)
print(response.choices[0].message)


# --------------------------------------------------------------
# 定义的具备查询航班功能的聊天机器人
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_number",
            "description": "根据始发地、目的地和日期，查询对应日期的航班号",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {"description": "出发地", "type": "string"},
                    "destination": {"description": "目的地", "type": "string"},
                    "date": {
                        "description": "日期",
                        "type": "string",
                    },
                },
                "required": ["departure", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "查询某航班在某日的票价",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {"description": "航班号", "type": "string"},
                    "date": {
                        "description": "日期",
                        "type": "string",
                    },
                },
                "required": ["flight_number", "date"],
            },
        },
    },
]

messages = []
messages.append(
    {
        "role": "user",
        "content": "帮我查询从2024年1月20日,从北京出发前往上海的航班,这趟航班的票价是多少？",
    }
)
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
print(response.choices[0].message)

messages.append(response.choices[0].message.model_dump())


# --------------------------------------------------------------
# Add a Function
# --------------------------------------------------------------
def get_flight_number(date: str, departure: str, destination: str):
    flight_number = {
        "北京": {
            "上海": "1234",
            "广州": "8321",
        },
        "上海": {
            "北京": "1233",
            "广州": "8123",
        },
    }
    return {"flight_number": flight_number[departure][destination]}


def get_ticket_price(date: str, flight_number: str):
    return {"ticket_price": "1000"}


# Use the LLM output to manually call the function
# The json.loads function converts the string to a Python object
output = response.choices[0].message

departure = json.loads(output.tool_calls[0].function.arguments).get("departure")
destination = json.loads(output.tool_calls[0].function.arguments).get("destination")
params = json.loads(output.tool_calls[0].function.arguments)
type(params)

print(departure)
print(destination)
print(params)

# Call the function with arguments

chosen_function = eval(output.tool_calls[0].function.name)
flight = chosen_function(**params)

print(flight)


# ----------------------------------------------------------------
# 定义处理 Function call 的函数
# ----------------------------------------------------------------


def parse_function_call(model_response, messages):
    # 处理函数调用结果，根据模型返回参数，调用对应的函数。
    # 调用函数返回结果后构造tool message，再次调用模型，将函数结果输入模型
    # 模型会将函数调用结果以自然语言格式返回给用户。
    if model_response.choices[0].message.tool_calls:
        tool_call = model_response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        function_result = {}
        if tool_call.function.name == "get_flight_number":
            function_result = get_flight_number(**json.loads(args))
        if tool_call.function.name == "get_ticket_price":
            function_result = get_ticket_price(**json.loads(args))
        messages.append(
            {
                "role": "tool",
                "content": f"{json.dumps(function_result)}",
                "tool_call_id": tool_call.id,
            }
        )
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=messages,
            tools=tools,
        )
        print(response.choices[0].message)
        messages.append(response.choices[0].message.model_dump())


#  查询北京到广州的航班：
messages = []

messages.append(
    {
        "role": "system",
        "content": "不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息",
    }
)
messages.append({"role": "user", "content": "帮我查询1月23日,北京到广州的航班"})

response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=messages,
    tools=tools,
)
print(response.choices[0].message)
messages.append(response.choices[0].message.model_dump())

parse_function_call(response, messages)


# 查询1234航班票价：
messages.append({"role": "user", "content": "这趟航班的价格是多少？"})
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=messages,
    tools=tools,
    tool_choice="auto",  # specify the function call
    # tool_choice={"type": "function", "function": {"name": "get_ticket_price"}},
)
print(response.choices[0].message)
messages.append(response.choices[0].message.model_dump())

parse_function_call(response, messages)


# --------------------------------------------------------------
# Make It Conversational With Langchain
# --------------------------------------------------------------


@tool
def get_flight_number(date: str, departure: str, destination: str):
    """根据始发地、目的地和日期，查询对应日期的航班号"""
    flight_number = {
        "北京": {
            "上海": "1234",
            "广州": "8321",
        },
        "上海": {
            "北京": "1233",
            "广州": "8123",
        },
    }
    return {"flight_number": flight_number[departure][destination]}


@tool
def get_ticket_price(date: str, flight_number: str):
    """查询某航班在某日的票价"""
    ticket_price = {
        "1234": "1000",
        "8321": "1200",
        "1233": "800",
        "8123": "1100",
    }
    return {"ticket_price": ticket_price[flight_number]}


tools = [get_flight_number, get_ticket_price]

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

llm_with_tools = llm.bind_tools(tools)

user_prompt = "帮我查询1月23日,北京到广州的航班,这趟航班的票价是多少？"

# Note that the zhupu model right now can NOT call multiple tools at once.

llm_with_tools.invoke(user_prompt)

llm_with_tools.invoke(user_prompt).tool_calls

# Extract the arguments from the first tool call
first_response = llm_with_tools.invoke(user_prompt)
# Convert AIMessage object to dictionary using __dict__
first_response_dict = first_response.__dict__
print(first_response_dict)


args = first_response_dict["additional_kwargs"]["tool_calls"][0]["function"][
    "arguments"
]

# extract the departure, destination, and parameters
departure = json.loads(args)["departure"]
destination = json.loads(args)["destination"]
params = json.loads(args)
print(departure)
print(destination)
print(params)

type(first_response)


# ----------------------------------------------------------------
# test
# ----------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

llm_with_tools = llm.bind_tools(tools)


from langchain_core.messages import HumanMessage, ToolMessage

query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
messages

llm_with_tools.invoke(messages)

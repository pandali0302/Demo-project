import os
from zhipuai import ZhipuAI
import json
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = ZhipuAI()

# use GLM's Function Call and code capabilities to perform simple data analysis


# ----------------------------------------------------------------
# Set up the function to execute the tool
# ----------------------------------------------------------------
# The tool execute_cleaned_code_from_string code is to execute the Python code output by the model.
def execute_cleaned_code_from_string(code_string: str = ""):
    import io
    from contextlib import redirect_stdout

    output_buffer = io.StringIO()
    try:
        code_object = compile(code_string, "<string>", "exec")
        with redirect_stdout(output_buffer):
            exec(code_object)
        return (
            output_buffer.getvalue()
            if output_buffer.getvalue()
            else "Code finished successfully!"
        )
    except Exception as e:
        error = "traceback: An error occurred: " + str(e)
        print(error)
        return error


def extract_function_and_execute(llm_output, messages):
    name = llm_output.choices[0].message.tool_calls[0].function.name
    params = json.loads(llm_output.choices[0].message.tool_calls[0].function.arguments)
    tool_call_id = llm_output.choices[0].message.tool_calls[0].id
    function_to_call = globals().get(name)
    if not function_to_call:
        raise ValueError(f"Function '{name}' not found")
    messages.append(
        {
            "role": "tool",
            "content": str(function_to_call(**params)),
            "tool_call_id": tool_call_id,
        }
    )
    return messages


tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_cleaned_code_from_string",
            "description": "python code execution tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_string": {
                        "type": "string",
                        "description": "Python executable code",
                    },
                },
                "required": ["code_string"],
            },
        },
    }
]

sys_prompt = """
You are a data analyst, you will have a code execution tool, you need to write a python script, the tool will execute your content and return the results, now please analyze my csv file.
I will provide you with some information about the csv, which is as follows:
{
     info: Changes in the urban-rural population ratio in major regions of the world from 1500 to 2050. The csv contains five columns, namely
     column info: Entity,Code,Year, Urban population (%) long-run with 2050 projections (OWID),Rural population (%) long-run with 2050 projections (OWID)
     path : 'data/urban-rural-population.csv'
}
Each column has some data, which you need to read through python code.
Now, please follow my requirements, write the code appropriately, and analyze my csv file.
I will provide you with the code to execute the tool, you just need to write the code according to my requirements.
All answers must be provided after querying the csv I provided. Your return must be executable python code and no other content.
Thinking step by step, here's my request, let's get started:
"""

question = "Read csv and draw the distribution of urban and rural population development in the United States"
messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": question},
]
response = client.chat.completions.create(
    model="glm-4",
    messages=messages,
    tools=tools,
    top_p=0.1,
    temperature=0.1,
    max_tokens=2000,
)
response
# python_code_str = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["code_string"]
# exec(python_code_str)


extract_function_and_execute(llm_output=response, messages=messages)


# ----------------------------------------------------------------
# 由于并不是所有的任务都能通过一次调用工具完成任务，因此，我们需要完善这个代码，
# 使得其能够在调用模型工具的时候获得反馈，如果代码无法执行，模型将获得报错的信息，
# 并重写代码，通过不断的优化，完成最终的任务。在这里，我们完善代码，
# 使其完成一个更复杂的任务，通过不管尝试，验证模型是否能完成任务。
# ----------------------------------------------------------------
questions = [
    "From 1972 to 2048, did more people in Russia live in cities or in rural areas? What is the rate of growth or decline, and when is the fastest growth or decline?",
    "Please draw the distribution of urban and rural population proportions in 2023 and 2030 for all the countries in the table, and summarize the trends.",
    "Compare the average growth rate of urbanized population in 'Colombia,Luxembourg and Macao' and draw a bar chart to tell me which country has the fastest growing proportion of urban population",
]

for question in questions:
    print("====================================================")
    print("Question:", question)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]

    number_try = 0
    while True:
        response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            top_p=0.1,
            tools=tools,
            temperature=0.1,
            max_tokens=2000,
        )

        if response.choices[0].finish_reason == "stop":
            print("Final answer for question:", question)
            print(response.choices[0].message.content)
            break
        elif response.choices[0].finish_reason == "tool_calls":
            number_try += 1
            if number_try > 10:
                print("Too many attempts, automatic stop for question:", question)
                break
            else:
                print(f"Try {number_try} times")
                print("====================================================\n\n")

            messages.append(
                {
                    "role": response.choices[0].message.role,
                    "tool_calls": [
                        {
                            "id": response.choices[0].message.tool_calls[0].id,
                            "type": "function",
                            "index": 0,
                            "function": {
                                "arguments": response.choices[0]
                                .message.tool_calls[0]
                                .function.arguments,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                            },
                        }
                    ],
                }
            )
            extract_function_and_execute(llm_output=response, messages=messages)

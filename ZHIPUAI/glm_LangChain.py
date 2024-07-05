# ----------------------------------------------------------------
# se the method of ChatOpenAI class to call ZhipuAI’s GLM-4 model
# ----------------------------------------------------------------
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key = os.getenv("ZHIPUAI_API_KEY")


llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
conversation.invoke({"question": "tell me a joke"})

# ----------------------------------------------------------------
# Use the method of ChatZhipuAI class from LangChain to call ZhipuAI’s GLM-4 model
# ----------------------------------------------------------------

# 和 ChatOpenAI 类一样，ChatZhipuAI 类也是一个可以调用 ZhipuAI 的 GLM-4 模型的类，我们可以使用 ChatZhipuAI来替换 ChatOpenAI 类，直接无缝对接到自己的项目中。这里展现了一个最基本的调用方法。
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a smart assistant, please reply to me smartly."),
    HumanMessage(
        content="There were 9 birds on the tree. The hunter shot and killed one. How many birds were there?"
    ),
]

llm = ChatZhipuAI(temperature=0.95, model="glm-4")

llm(messages).content


# ----------------------------------------------------------------
# Use Langchain to call your own fine-tuned model from ZhipuAI server
# ----------------------------------------------------------------
fine_tuned_model = ChatZhipuAI(
    temperature=0.2, model_name="ft:glm-4:advertise_gen:sg59pfxk"
)


# ----------------------------------------------------------------
# 使用 Langchain 来完成一个简单的 Function Call 调用
# Calling GLM-4 with LangChain AgentExecutor
# ----------------------------------------------------------------
# refer to the solution of Langchain official tutorial and just replace ChatOpenAI with ChatZhipuAI. Except for the model loading method, in Agent There is absolutely no need to make any code modifications in the code.
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/react-chat-json")
agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

agent_executor.invoke({"input": "what is LangChain?"})

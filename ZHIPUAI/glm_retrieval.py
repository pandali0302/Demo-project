# ----------------------------------------------------------------
# Use the GLM-4 model combined with the knowledge base function of Zhipu AI to answer questions
# ----------------------------------------------------------------
import os
from zhipuai import ZhipuAI

os.environ["ZHIPUAI_API_KEY"] = "your api key here"
client = ZhipuAI()

prompt_template = """
从文档
{{knowledge}}
中找问题
{{question}}
的答案，
找到答案就仅使用文档语句回答，找不到答案就用自身知识回答并告诉用户该信息不是来自文档。
不要复述问题，直接开始回答。
"""
knowledge_id = 1748261416734965760  ## my knowledge base id
response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "大数据专业未来的升学情况"},
    ],
    tools=[
        {
            "type": "retrieval",
            "retrieval": {
                "knowledge_id": knowledge_id,
                "prompt_template": prompt_template,
            },
        }
    ],
    stream=False,
)

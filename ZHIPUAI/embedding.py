# ----------------------------------------------------------------
# UsingOpenAI SDK to call Embedding Model
# ----------------------------------------------------------------
from openai import OpenAI

client = OpenAI(
    api_key="your api key", base_url="https://open.bigmodel.cn/api/paas/v4/"
)
response_embedding = client.embeddings.create(
    model="embedding-2",
    input="你好",
    # these parameters are not available
    # input=["你好", "你是谁"],
    # dimensions=1024,
    # encoding_format="float",
)
response_embedding.data[0].embedding[:10]  # only show the first 10


# ----------------------------------------------------------------
# Using ZhipuAI Embedding API to get the embedding of a text
# https://github.com/MetaGLM/glm-cookbook/blob/99c62a65f0d81808092ada57dba1e891451dce44/basic/glm_embedding_pysdk.ipynb
# ----------------------------------------------------------------
#!pip install faiss-cpu scikit-learn scipy
import os
from zhipuai import ZhipuAI

os.environ["ZHIPUAI_API_KEY"] = "your api key"
client = ZhipuAI()

embedding_text = "hello world"
response = client.embeddings.create(
    model="embedding-2",
    input=embedding_text,
)

response.usage.total_tokens  # The number of tokens used

response.data[0].embedding  # The support behind Embedding is a 1024


# ----------------------------------------------------------------
# Use vector database and search
# ----------------------------------------------------------------
import numpy as np
import faiss

embedding_text = """
Multimodal Agent AI systems have many applications. In addition to interactive AI, grounded multimodal models could help drive content generation for bots and AI agents, and assist in productivity applications, helping to re-play, paraphrase, action prediction or synthesize 3D or 2D scenario. Fundamental advances in agent AI help contribute towards these goals and many would benefit from a greater understanding of how to model embodied and empathetic in a simulate reality or a real world. Arguably many of these applications could have positive benefits.

However, this technology could also be used by bad actors. Agent AI systems that generate content can be used to manipulate or deceive people. Therefore, it is very important that this technology is developed in accordance with responsible AI guidelines. For example, explicitly communicating to users that content is generated by an AI system and providing the user with controls in order to customize such a system. It is possible the Agent AI could be used to develop new methods to detect manipulative content - partly because it is rich with hallucination performance of large foundation model - and thus help address another real world problem.

For examples, 1) in health topic, ethical deployment of LLM and VLM agents, especially in sensitive domains like healthcare, is paramount. AI agents trained on biased data could potentially worsen health disparities by providing inaccurate diagnoses for underrepresented groups. Moreover, the handling of sensitive patient data by AI agents raises significant privacy and confidentiality concerns. 2) In the gaming industry, AI agents could transform the role of developers, shifting their focus from scripting non-player characters to refining agent learning processes. Similarly, adaptive robotic systems could redefine manufacturing roles, necessitating new skill sets rather than replacing human workers. Navigating these transitions responsibly is vital to minimize potential socio-economic disruptions.

Furthermore, the agent AI focuses on learning collaboration policy in simulation and there is some risk if directly applying the policy to the real world due to the distribution shift. Robust testing and continual safety monitoring mechanisms should be put in place to minimize risks of unpredictable behaviors in real-world scenarios. Our “VideoAnalytica" dataset is collected from the Internet and considering which is not a fully representative source, so we already go through-ed the ethical review and legal process from both Microsoft and University Washington. Be that as it may, we also need to understand biases that might exist in this corpus. Data distributions can be characterized in many ways. In this workshop, we have captured how the agent level distribution in our dataset is different from other existing datasets. However, there is much more than could be included in a single dataset or workshop. We would argue that there is a need for more approaches or discussion linked to real tasks or topics and that by making these data or system available.

We will dedicate a segment of our project to discussing these ethical issues, exploring potential mitigation strategies, and deploying a responsible multi-modal AI agent. We hope to help more researchers answer these questions together via this paper.

"""

chunk_size = 150
chunks = [
    embedding_text[i : i + chunk_size]
    for i in range(0, len(embedding_text), chunk_size)
]
chunks


from sklearn.preprocessing import normalize
import numpy as np
import faiss

embeddings = []
for chunk in chunks:
    response = client.embeddings.create(
        model="embedding-2",
        input=chunk,
    )
    embeddings.append(response.data[0].embedding)
normalized_embeddings = normalize(np.array(embeddings).astype("float32"))
d = 1024
index = faiss.IndexFlatIP(d)
index.add(normalized_embeddings)

n_vectors = index.ntotal

n_vectors

# ----------------------------------------------------------------
# 我们可以使用向量数据库进行检索。下面代码实现了一个名为match_text的函数，
# 其目的是在一个文本集合中找到与给定输入文本最相似的文本块。
# 其中 k是要返回的相似文本块的数量。
# ----------------------------------------------------------------
from sklearn.preprocessing import normalize


def match_text(input_text, index, chunks, k=2):
    k = min(k, len(chunks))

    response = client.embeddings.create(
        model="embedding-2",
        input=input_text,
    )
    input_embedding = response.data[0].embedding
    input_embedding = normalize(np.array([input_embedding]).astype("float32"))

    distances, indices = index.search(input_embedding, k)

    for i, idx in enumerate(indices[0]):
        print(f"similarity: {distances[0][i]:.4f}\nmatching text: \n{chunks[idx]}\n")


input_text = "VideoAnalytica dataset"

matched_texts = match_text(input_text=input_text, index=index, chunks=chunks, k=2)

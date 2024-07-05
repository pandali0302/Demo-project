# ----------------------------------------------------------------
# use BilibiliLoader from LangChain to load the transcript of a Bilibili video
# ----------------------------------------------------------------
from langchain_community.document_loaders import BiliBiliLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import textwrap
import os
import gradio as gr

_ = load_dotenv(find_dotenv())
api_key = os.getenv("ZHIPUAI_API_KEY")
SESSDATA = os.getenv("SESSDATA")
BILI_JCT = os.getenv("BILI_JCT")
BUVID3 = os.getenv("BUVID3")

# Set TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_db_from_Bilibili_video_url(video_url, query, k=4):
    loader = BiliBiliLoader(
        video_urls=[video_url],
        sessdata=SESSDATA,
        bili_jct=BILI_JCT,
        buvid3=BUVID3,
    )

    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # embeddings = OllamaEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4",
        openai_api_key=api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about bilibili videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=llm, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response, docs


# Example usage:
# video_url = "https://www.bilibili.com/video/BV14f421q7H9/?spm_id_from=333.788.playrecommendByOp.0&vd_source=f187bdb853f3690201169c42c99729e6"
# query = "请用中文回答桃子变软的小技巧是什么？"
# response, docs = create_db_from_Bilibili_video_url(video_url, query)

# print(textwrap.fill(response, width=50))

# ----------------------------------------------------------------
# create web demo using Gradio with title and description and two input fields: video_url and query based on the above code
# ----------------------------------------------------------------
import gradio as gr


def get_response(video_url, query):
    response, _ = create_db_from_Bilibili_video_url(video_url, query)
    return response


video_url_input = gr.Textbox(label="Enter video URL:")
query_input = gr.Textbox(label="Enter your question:")
outputs = gr.Textbox(label="Answer:")

title = "Bilibili Video QA"
description = "Ask questions about Bilibili videos based on their transcripts"

gr.close_all()
gr.Interface(
    fn=get_response,
    inputs=[video_url_input, query_input],
    outputs=outputs,
    title=title,
    description=description,
    allow_flagging="never",
).launch()

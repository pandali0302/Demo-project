from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # by default, uses llama2.
    return embeddings

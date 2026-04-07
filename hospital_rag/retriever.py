from functools import cache

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from hospital_rag.config import settings


@cache
def get_retriever() -> VectorStoreRetriever:
    vector_db = Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=OpenAIEmbeddings(api_key=settings.openai_api_key),
    )
    return vector_db.as_retriever(search_kwargs={"k": settings.retriever_k})

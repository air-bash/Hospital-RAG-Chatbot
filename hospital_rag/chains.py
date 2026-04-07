from functools import cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from hospital_rag.config import settings
from hospital_rag.retriever import get_retriever

_REVIEW_SYSTEM_PROMPT = """\
Your job is to use patient reviews to answer questions about their experience at a hospital.
Use the following context to answer questions. Be as detailed as possible, but don't make up
any information that's not from the context. If you don't know an answer, say you don't know.

{context}"""

_review_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(_REVIEW_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}"),
])


@cache
def get_review_chain() -> Runnable:
    return (
        {"context": get_retriever(), "question": RunnablePassthrough()}
        | _review_prompt
        | ChatOpenAI(model=settings.model_name, temperature=0, api_key=settings.openai_api_key)
        | StrOutputParser()
    )

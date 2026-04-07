from functools import cache

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from hospital_rag.config import settings
from hospital_rag.tools import reviews_tool, waits_tool

_SYSTEM_PROMPT = (
    "You are a helpful hospital assistant. "
    "Answer questions about patient reviews and current wait times."
)


@cache
def get_agent():
    return create_agent(
        model=ChatOpenAI(model=settings.model_name, temperature=0, api_key=settings.openai_api_key),
        tools=[reviews_tool, waits_tool],
        system_prompt=_SYSTEM_PROMPT,
        debug=settings.debug,
    )

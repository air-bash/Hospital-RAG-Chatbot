import random
import time

from langchain_core.tools import tool

from hospital_rag.chains import get_review_chain

_KNOWN_HOSPITALS = frozenset({"A", "B", "C", "D"})


@tool
def reviews_tool(question: str) -> str:
    """Useful when you need to answer questions about patient reviews or experiences at
    the hospital. Not useful for answering questions about specific visit details such as
    payer, billing, treatment, diagnosis, chief complaint, hospital, or physician information.
    Pass the entire question as input to the tool. For instance, if the question is
    "What do patients think about the triage system?", the input should be
    "What do patients think about the triage system?"
    """
    return get_review_chain().invoke(question)


@tool
def waits_tool(hospital: str) -> int | str:
    """Use when asked about current wait times at a specific hospital. This tool can only
    get the current wait time at a hospital and does not have any information about aggregate
    or historical wait times. Returns wait times in minutes. Do not pass the word "hospital"
    as input, only the hospital name itself. For instance, if the question is
    "What is the wait time at hospital A?", the input should be "A".
    """
    if hospital not in _KNOWN_HOSPITALS:
        return f"Hospital {hospital} does not exist"
    time.sleep(1)  # TODO: replace with real hospital API call
    return random.randint(0, 10000)

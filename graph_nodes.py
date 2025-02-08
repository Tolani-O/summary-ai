from typing import List, Dict, TypedDict, Annotated, Literal
import operator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from helper_functions import length_function
import re
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
input_token_max = 100000
summary_token_max = 10000

# LLM setup
summary_agent = ChatOllama(base_url="http://127.0.0.1:11434", model="llama3.2:3b-instruct-q5_K_M", num_ctx=10000, num_predict=summary_token_max, verbose=True)
validate_agent = ChatOllama(base_url="http://127.0.0.1:11434", model="deepseek-r1:1.5b", num_ctx=100000, num_predict=summary_token_max, verbose=True)
refine_agent = ChatOllama(base_url="http://127.0.0.1:11434", model="llama3.2:3b-instruct-q5_K_M", num_ctx=100000, num_predict=summary_token_max, verbose=True)

# Prompts
map_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled document summarizer specialized in condensing a complex section of a document into clear, concise, single paragraph summaries."),
    ("human",
     "Please read the following document section and provide a concise summary highlighting the main ideas and key details in only one paragraph. The document section starts here:\n\n{section}"),
    ("ai", "Section Summary: ")
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled document summarizer, specialized in creating concise, structured, bullet-point summaries from large document content."),
    ("human",
     "The following document needs to be summarized into a clear and concise bullet-point list of main ideas. Summarize the document here:\n\n{summaries}"),
    ("ai", "Document Bullet-Point Summary:")
])

validate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled summary verifier, adept at verifying whether or not a one document is an accurate summary of another, larger document. "),
    ("human",
     "Please verify if the following summary accurately represents the main ideas and key details of the original content. If there are any inaccuracies or hallucinations, please correct them. Respond with an accuracy score in the format 'Summary accuracy: \d\d\%.' The original content starts here:\n\n```\n{original_content}\n```\n\nThe summary to verify is:\n\n```\n{summary}\n```"),
    ("ai", "Summary accuracy: ")
])

refine_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled summary refiner, adept at modifying a summary of an original document to more accurately reflect the contents of the original document, based on suggested feedback. "),
    ("human",
     "Please modify the following summary to more accurately represent the main ideas and key details of the original content, incorporating any suggestions given in the suggested feedback. If there are any inaccuracies or hallucinations, please correct them. Your goal is to score above the current score of {accuracy}% summary accuracy. The original content starts here:\n\n```\n{original_content}\n```\n\nThe summary to modify starts here:\n\n```\n{summary}\n```\n\n. The suggested feedback starts here:\n\n```\n{suggested_feedback}\n```"),
    ("ai", "Modified Bullet-Point Summary: ")
])


### Graph state object
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[str]
    final_summary: str
    needs_refinement: bool
    validation_result: str
    summary_accuracy: float


### Graph nodes: These update the state object
async def generate_summary(content: Dict[str, str]):
    prompt = map_prompt.invoke(content)
    response = await summary_agent.ainvoke(prompt)
    split_content = response.content.strip().replace("\n\n", "\n")
    return {"summaries": [split_content]}

def collect_summaries(state: OverallState):
    return {"collapsed_summaries": [summary for summary in state["summaries"]]}

async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs([Document(summary) for summary in state["collapsed_summaries"]], length_function, input_token_max, llm=summary_agent)
    results = []
    for doc_list in doc_lists:
        results.append((await acollapse_docs(doc_list, _reduce)).page_content)
    return {"collapsed_summaries": results}

async def generate_final_summary(state: OverallState):
    response = await _reduce([Document(content) for content in state["collapsed_summaries"]])
    return {"final_summary": response}

# Node helper function to reduce summaries
async def _reduce(docs: List[Document]) -> str:
    summaries_text = "\n".join([doc.page_content for doc in docs])
    prompt = reduce_prompt.invoke({"summaries": summaries_text})
    response = await summary_agent.ainvoke(prompt)
    summary = response.content
    return summary

async def validate_summary_node(state: OverallState):
    original_content = "\n\n".join(state["collapsed_summaries"])
    summary = state["final_summary"]
    prompt = validate_prompt.invoke({"original_content": original_content, "summary": summary})
    context_length = length_function([validate_prompt.format(original_content=original_content, summary=summary)], validate_agent)
    response = await validate_agent.ainvoke(prompt)
    validation_result = response.content
    # Determine if refinement is needed based on validation result
    match = re.search(r"(?:summary accuracy: )?(\d+(?:\.\d+)?)%", validation_result, re.IGNORECASE)
    while not match:
        logger.error(f"Validation result doesn't contain an accuracy percentage: \n{validation_result}")
        logger.error("Retrying validation.")
        return await validate_summary_node(state)  # Retry validation
    summary_accuracy = float(match.group(1))
    needs_refinement = summary_accuracy < 90
    suggested_feedback = validation_result.split("</think>")[-1].strip()
    return {"needs_refinement": needs_refinement, "summary_accuracy": summary_accuracy, "suggested_feedback": suggested_feedback, "final_summary": summary}

async def refine_summary_node(state: OverallState):
    original_content = "\n".join(state["contents"])
    summary = state["final_summary"]
    validation_result = state["validation_result"]
    prompt = refine_prompt.invoke({"original_content": original_content, "summary": summary, "validation_result": validation_result, "accuracy": state["summary_accuracy"]})
    context_length = length_function([validate_prompt.format(original_content=original_content, summary=summary, validation_result=validation_result, accuracy=state["summary_accuracy"])], refine_agent)
    response = await refine_agent.ainvoke(prompt)
    refined_summary = response.content
    return {"final_summary": refined_summary}


### Edge routing functions: These route the edges
def map_summaries(state: OverallState):
    return [Send("generate_summary", {"section": content}) for content in state["contents"]]

def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"], summary_agent)
    if num_tokens > input_token_max:
        return "collapse_summaries"
    return "generate_final_summary"

def decide_refinement(state: OverallState) -> str:
    if state["needs_refinement"]:
        return "refine_summary"
    return END


### Graph construction
def build_single_agent_graph():
    graph = StateGraph(OverallState)
    # Define nodes
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Define edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    return graph.compile()

def build_multi_agent_graph():
    graph = StateGraph(OverallState)
    # Define nodes
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)
    graph.add_node("validate_summary", validate_summary_node)
    graph.add_node("refine_summary", refine_summary_node)

    # Define edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", "validate_summary")
    graph.add_conditional_edges("validate_summary", decide_refinement)
    graph.add_edge("refine_summary", "validate_summary")

    return graph.compile()
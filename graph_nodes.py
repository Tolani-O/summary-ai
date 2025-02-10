from typing import List, TypedDict, Annotated, Literal
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents.reduce import split_list_of_docs
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from helper_functions import length_function
import re
import logging
import asyncio
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
input_token_max = 6000 # So the model does not get lost
summary_token_max = 10000

# Check if running in Docker
def is_running_in_docker():
    # Check for the presence of the Docker environment variable
    return os.path.exists('/.dockerenv') or os.path.isfile('/proc/self/cgroup') and 'docker' in open('/proc/self/cgroup').read()

# Set base URL based on the environment
if is_running_in_docker():
    base_url = "http://host.docker.internal:11434"  # Docker base URL
else:
    base_url = "http://localhost:11434"  # Local base URL

# Initialize agents with the determined base URL
section_summary_agent = ChatOllama(base_url=base_url, model="llama3.2:3b-instruct-q5_K_M", num_ctx=10000, num_predict=summary_token_max, verbose=True)
document_summary_agent = ChatOllama(base_url=base_url, model="llama3.2:3b-instruct-q5_K_M", num_ctx=100000, num_predict=summary_token_max, verbose=True)
validate_agent = ChatOllama(base_url=base_url, model="deepseek-r1:1.5b", num_ctx=100000, num_predict=summary_token_max, verbose=True)

# Prompts
map_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled document summarizer specialized in condensing a complex section of a document into clear, concise, single paragraph summaries."),
    ("human",
     "Please read the following document section and provide a concise summary highlighting the main ideas and key details in only one paragraph. The document section starts here:\n\n{section}"),
    ("ai", "Section Summary: ")
])

output_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled document summarizer, specialized in creating concise, structured, bullet-point summaries from large document content."),
    ("human",
     "The following document needs to be summarized into a clear and concise bullet-point list of main ideas. Summarize the document here:\n\n{document}"),
    ("ai", "Document Bullet-Point Summary:")
])

validate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled summary verifier, adept at verifying whether or not a one document is an accurate summary of another, larger document. "),
    ("human",
     "Please verify if the following summary accurately represents the main ideas and key details of the original content. If there are any inaccuracies or hallucinations, please correct them. Respond with an summary accuracy as a percentage (%).' The original content starts here:\n\n```\n{original_content}\n```\n\nThe summary to verify is:\n\n```\n{summary}\n```"),
    ("ai", "Summary accuracy: ")
])

refine_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly skilled summary refiner, adept at modifying a summary of an original document to more accurately reflect the contents of the original document, based on suggested feedback. "),
    ("human",
     "Please modify the following summary to more accurately represent the main ideas and key details of the original content as a bullet point list. Incorporate any suggestions given in the suggested feedback. If there are any inaccuracies, please correct them. Your goal is to score above the current score of {accuracy}% summary accuracy. The original content starts here:\n\n```\n{original_content}\n```\n\nThe summary to modify starts here:\n\n```\n{summary}\n```\n\n. The suggested feedback starts here:\n\n```\n{suggested_feedback}\n```"),
    ("ai", "Modified Bullet-Point Summary: ")
])


### Graph state object
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[str]
    final_summary: str
    needs_refinement: bool
    suggested_feedback: str
    summary_accuracy: float


### Graph nodes: These update the state object
async def generate_summary(content: str):
    prompt = map_prompt.invoke({"section": content})
    response = await section_summary_agent.ainvoke(prompt)
    cleaned_content = response.content.strip().replace("\n\n", "\n")
    return {"summaries": [cleaned_content]}

def collect_summaries(state: OverallState):
    return {"collapsed_summaries": [summary for summary in state["summaries"]]}

async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, input_token_max, llm=section_summary_agent)
    results = await asyncio.gather(*[
        section_summary_agent.ainvoke(map_prompt.invoke({"section": "\n\n".join(doc_list)}))
        for doc_list in doc_lists
    ])
    cleaned_contents = [response.content.strip().replace("\n\n", "\n") for response in results]
    return {"collapsed_summaries": cleaned_contents}

async def generate_final_summary(state: OverallState):
    prompt = output_prompt.invoke({"document": "\n\n".join(state["collapsed_summaries"])})
    response = await document_summary_agent.ainvoke(prompt)
    cleaned_content = response.content.strip().replace("\n\n", "\n")
    return {"final_summary": cleaned_content}

async def validate_summary(state: OverallState):
    prompt = validate_prompt.invoke({"original_content": "\n\n".join(state["collapsed_summaries"]),
                                     "summary": state["final_summary"]})
    # context_length = length_function([validate_prompt.format(original_content=original_content, summary=summary)], validate_agent)
    response = await validate_agent.ainvoke(prompt)
    validation_result = response.content
    logger.info(f"Validation result:\n{validation_result}")
    # Determine if refinement is needed based on validation result
    match = re.search(r"(?:summary accuracy: )?(\d+(?:\.\d+)?)%", validation_result, re.IGNORECASE)
    while not match:
        logger.error(f"Validation result doesn't contain an accuracy percentage. Retrying validation.")
        return await validate_summary(state)  # Retry validation
    summary_accuracy = float(match.group(1))
    needs_refinement = summary_accuracy < 90
    suggested_feedback = validation_result.split("</think>")[-1].strip().replace("\n\n", "\n")
    return {"needs_refinement": needs_refinement, "summary_accuracy": summary_accuracy, "suggested_feedback": suggested_feedback, "final_summary": state["final_summary"]}

async def refine_summary(state: OverallState):
    prompt = refine_prompt.invoke({"original_content": "\n\n".join(state["collapsed_summaries"]),
                                   "summary": state["final_summary"],
                                   "suggested_feedback": state["suggested_feedback"],
                                   "accuracy": state["summary_accuracy"]})
    # context_length = length_function([validate_prompt.format(original_content=original_content, summary=summary, suggested_feedback=suggested_feedback, accuracy=state["summary_accuracy"])], refine_agent)
    response = await document_summary_agent.ainvoke(prompt)
    cleaned_content = response.content.strip().replace("\n\n", "\n")
    return {"final_summary": cleaned_content}


### Edge routing functions: These route the edges
def map_summaries(state: OverallState):
    return [Send("generate_summary", content) for content in state["contents"]]

def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function([output_prompt.format(document="\n\n".join(state["collapsed_summaries"]))], section_summary_agent)
    print(f"Number of tokens: {num_tokens}, Max: {input_token_max}")
    if num_tokens > input_token_max:
        return "collapse_summaries"
    return "generate_final_summary"

def decide_refinement(state: OverallState) -> Literal["refine_summary", "__end__"]:
    if state["needs_refinement"]:
        return "refine_summary"
    return END


### Graph construction
def build_single_agent_graph():
    graph = StateGraph(OverallState)
    # Define nodes
    graph.add_node(generate_summary)
    graph.add_node(collect_summaries)
    graph.add_node(collapse_summaries)
    graph.add_node(generate_final_summary)

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
    graph.add_node(generate_summary)
    graph.add_node(collect_summaries)
    graph.add_node(collapse_summaries)
    graph.add_node(generate_final_summary)
    graph.add_node(validate_summary)
    graph.add_node(refine_summary)

    # Define edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", "validate_summary")
    graph.add_conditional_edges("validate_summary", decide_refinement)
    graph.add_edge("refine_summary", "validate_summary")

    return graph.compile()
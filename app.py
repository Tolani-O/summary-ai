from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_ollama.llms import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import fitz  # PyMuPDF for PDF processing
import logging

app = FastAPI()

llm = OllamaLLM(base_url="http://127.0.0.1:11434", model="phi3", num_ctx=15000, num_gpu=10, num_predict=600)

# Create a ChatPromptTemplate for summarizing a section
section_system_msg = SystemMessagePromptTemplate.from_template(
    "You are a highly skilled summarizer specialized in condensing complex documents into clear, concise summaries."
)
section_human_msg = HumanMessagePromptTemplate.from_template(
    "Please read the following section and provide a concise summary highlighting the main ideas and key details:\n\n{section}"
)
section_ai_msg = AIMessagePromptTemplate.from_template("Summary:")
section_chat_prompt = ChatPromptTemplate.from_messages([
    section_system_msg,
    section_human_msg,
    section_ai_msg
])

# Create a ChatPromptTemplate for combining summaries
combine_system_msg = SystemMessagePromptTemplate.from_template(
    "You are an expert synthesizer adept at merging multiple pieces of information into a coherent, concise format."
)
combine_human_msg = HumanMessagePromptTemplate.from_template(
    "Given the following section summaries, produce a concise bullet-point list that captures the key points of the document:\n\n{summaries}"
)
combine_ai_msg = AIMessagePromptTemplate.from_template("Bullet-Point Summary:")
combine_chat_prompt = ChatPromptTemplate.from_messages([
    combine_system_msg,
    combine_human_msg,
    combine_ai_msg
])

# Create LLMChain instances using the chat prompt templates
section_chain = section_chat_prompt | llm
combine_chain = combine_chat_prompt | llm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    path: str
    mode: str


def parse_sections(pdf_path: str, footer_threshold: float = 700, header_threshold: float = 120) -> dict[str, str]:
    """Remove headers and footers from a PDF document and return the cleaned text."""
    doc = fitz.open(pdf_path)

    content_topics = set()
    redundant_topics = {'Summary', 'Bibliographic Remarks', 'Examples'}
    full_content = {}
    current_section = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").lower()
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            # Skip footers, headers, or non-text blocks
            if y0 > footer_threshold or y1 < header_threshold or block_type != 0:
                continue
            block_lines = tuple(text.splitlines())
            if "contents" in page_text:
                for line in block_lines:
                    if not line.strip().replace('.', '').isdigit():
                        content_topics.add(line)
            elif (len(block_lines) == 2) and (block_lines[1] not in redundant_topics) and (block_lines[1] in content_topics):
                if block_lines not in full_content:
                    full_content[block_lines] = ""
                    current_section = block_lines
            if current_section in full_content:
                cleaned_text = text.replace("\n", " ").strip()
                full_content[current_section] += f"{cleaned_text}\n"
    return full_content


@app.post("/summarize", response_class=PlainTextResponse)
async def summarize_single(request: TextRequest):
    if request.mode not in ("single", "multi"):
        raise HTTPException(status_code=400, detail="Mode must be 'single' or 'multi'.")
    elif not request.path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for summarization.")
    try:
        logger.info(f"Starting {request.mode}-agent summarization")
        logger.info(f"Parsing {request.path}...")
        text_sections = parse_sections(request.path)
        num_sections = len(text_sections)
        logger.info(f"Number of sections: {num_sections}")
        if request.mode == "single":
            # For a single-agent summarization, use a prebuilt chain (e.g., map-reduce)
            docs = [Document(page_content=section) for section in text_sections.values()]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = (await chain.ainvoke(docs))["output_text"]
        else:
            # Create a list of tasks with their original ordering.
            titles = list(text_sections.keys())
            sections = list(text_sections.values())

            section_summaries = section_chain.abatch_as_completed([{"section": content} for content in sections])
            ordered_summaries = [None] * num_sections
            num_completed = 1
            async for idx, summary in section_summaries:
                title = titles[idx]
                logger.info(f"Task {idx} for section '{title}' completed with summary:\n{summary}\n")
                logger.info(f"Completed {num_completed} tasks out of {num_sections}\n\n")
                ordered_summaries[idx] = f"{title}\n{summary}"
                num_completed += 1
            # Combine the individual summaries using the combine chain
            combined_summaries = "\n\n".join(ordered_summaries)
            summary = await combine_chain.ainvoke({"summaries": combined_summaries})
        logger.info(f"Summarization completed. Output:\n{summary}")
        # Writing the summary to a text file
        with open("sample_text/summary.txt", "w", encoding="utf-8") as file:
            file.write(summary)
        return "Summary:\n" + summary
    except Exception as e:
        logger.error(f"Error in {request.mode}-agent summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

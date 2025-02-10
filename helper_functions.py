import fitz  # PyMuPDF for PDF processing
from typing import List

# PDF parsing function
def parse_sections(pdf_path: str, footer_threshold: float = 700, header_threshold: float = 120) -> dict[str, str]:
    """Remove headers and footers from a PDF document and return the cleaned text."""
    doc = fitz.open(pdf_path)

    content_topics = set()
    redundant_topics = {'Summary', 'Bibliographic Remarks', 'Exercises'}
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
            elif len(block_lines) == 2:
                if block_lines[1] in redundant_topics:
                    current_section = ""
                elif block_lines[1] in content_topics:
                    current_section = " ".join(block_lines)
                    if current_section not in full_content:
                        full_content[current_section] = ""
            if current_section in full_content:
                cleaned_text = text.replace("\n", " ").strip()
                full_content[current_section] += f"{cleaned_text}\n"

        # if page_num == 31: break  # Only for debugging

    return full_content


# Helper function to calculate token length
def length_function(documents: List[str], llm) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc) for doc in documents)
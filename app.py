from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from helper_functions import parse_sections
from graph_nodes import build_single_agent_graph, build_multi_agent_graph
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# FastAPI endpoint
class TextRequest(BaseModel):
    path: str
    mode: str

@app.post("/summarize", response_class=PlainTextResponse)
async def summarize(request: TextRequest):
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

        # Choose the appropriate graph based on the mode
        if request.mode == "single":
            app_graph = build_single_agent_graph()
        else:
            app_graph = build_multi_agent_graph()
        num_completed = 1
        final_state = None

        # Stream the graph execution and capture the final state
        async for step in app_graph.astream(
                {"contents": list(text_sections.values())},
                {"recursion_limit": 10}, stream_mode="updates"
        ):
            print(step)
            print(f"Completed {num_completed} tasks out of {num_sections}\n\n")
            num_completed += 1
            final_state = step  # Capture the latest state update

        # Ensure the final state is available
        if final_state is None:
            raise HTTPException(status_code=500, detail="Graph execution did not produce a final state.")

        # Extract the final summary from the final state
        summary = final_state.get("validate_summary", {}).get("final_summary")
        if summary is None:
            raise HTTPException(status_code=500, detail="Final summary not found in the graph state.")

        logger.info(f"Summarization completed. Output:\n{summary}")
        with open("sample_text/summary.txt", "w", encoding="utf-8") as file:
            file.write(summary)
        return "Summary:\n" + summary
    except Exception as e:
        logger.error(f"Error in {request.mode}-agent summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
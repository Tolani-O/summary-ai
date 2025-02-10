from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from graph_nodes import build_agent_graph
import logging
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# FastAPI endpoint
class TextRequest(BaseModel):
    input_path: str
    mode: str = "single"
    log_output_path: str
    summary_output_path: str
    recursion_limit: int = 10
    validation_threshold: float

@app.post("/summarize", response_class=PlainTextResponse)
async def summarize(request: TextRequest):
    if request.mode not in ("single", "multi"):
        raise HTTPException(status_code=400, detail="Mode must be 'single' or 'multi'.")
    elif not request.input_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for summarization.")

    try:
        # Choose the appropriate graph based on the mode
        app_graph = build_agent_graph()
        if request.mode == "single":
            request.validation_threshold = 0  # Disable validation for single-agent mode

        async def event_stream():
            # Ensure output paths' directories exist
            os.makedirs(os.path.dirname(request.log_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(request.summary_output_path), exist_ok=True)
            final_state = None
            # Open the output file in write mode
            with open(request.log_output_path, "w") as output_file:
                print(f"Starting {request.mode}-agent summarization\n")
                yield f"Starting {request.mode}-agent summarization\n\n"
                output_file.write(f"Starting {request.mode}-agent summarization\n\n")
                output_file.flush()
                print(f"Parsing {request.input_path}...\n")
                yield f"Parsing {request.input_path}...\n\n"
                output_file.write(f"Parsing {request.input_path}...\n\n")
                output_file.flush()

                try:
                    # Stream the graph execution and capture the final state
                    async for step in app_graph.astream({
                        "input_path": request.input_path,
                        "validation_threshold": request.validation_threshold
                    },
                    {
                        "recursion_limit": request.recursion_limit
                    },  stream_mode="updates"):
                        # Stream the output to the client
                        output_line = f"Step output:\n{str(step)}"  # Ensure it's a string
                        print(f"{output_line}\n")
                        yield f"Step: {list(step.keys())[0]}\n\n"
                        output_file.write(f"{output_line}\n\n")  # Write to the output file
                        output_file.flush()
                        final_state = list(step.values())[0]  # Capture the latest state update
                except RecursionError as e:
                    # If recursion limit is hit, proceed with the state computed so far
                    print(f"Recursion limit exceeded: {str(e)}. Proceeding with the current state.")
                    yield f"Recursion limit exceeded: {str(e)}. Proceeding with the current state.\n"
                    output_file.write(f"Recursion limit exceeded: {str(e)}. Proceeding with the current state.\n\n")
                    output_file.flush()

                # Ensure the final state is available after streaming
                if final_state is None:
                    error_message = "Error: Graph execution did not produce a final state. Please try again."
                    print(error_message)
                    yield f"{error_message}\n"
                    output_file.write(f"{error_message}\n\n")  # Write to the output file
                    output_file.flush()
                    return

                # Process the final state to get the summary
                summary = final_state.get("final_summary")
                if summary is None:
                    error_message = "Error: Final summary not found in the final state. Please try again."
                    print(error_message)
                    yield f"{error_message}\n"
                    output_file.write(f"{error_message}\n\n")  # Write to the output file
                    output_file.flush()
                    return

            summary_message = f"Summary completed. Output:\n\n{str(summary)}\n"  # Ensure it's a string
            print(summary_message)
            yield f"{summary_message}\n"
            with open(request.summary_output_path, "w") as output_file:
                output_file.write(f"{summary_message}\n\n") # Write to the output file
                output_file.flush()
        return StreamingResponse(event_stream(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error in {request.mode}-agent summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Summary Agent Docker Setup

This guide explains how to run the Summary Agent Docker container.

## Prerequisites

1. Install Docker Desktop
2. Install Ollama
3. Have the phi3 model downloaded (`ollama pull phi3`)

## Setup Steps

1. Start Ollama server in a terminal:
```bash
ollama serve
```

2. Start FastAPI in a terminal
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Extract the text of the input document and save as a json file
```bash
python payload.py sample_text/understanding-machine-learning-theory-algorithms.pdf
```

2. Build the Docker image:
```bash
docker build -t summary-agent .
```

3. Run the Docker container:
```bash
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway summary-agent
```

## Testing

In a new terminal, run the test script:
```bash
chmod +x test.sh  # Make script executable (first time only)
./test.sh
```

## Troubleshooting

If you encounter issues:
1. Ensure Ollama is running (`ollama serve`)
2. Check Docker container logs: `docker logs $(docker ps -q)`
3. Verify container is running: `docker ps`
4. Stop existing containers if needed: `docker stop $(docker ps -aq)`

## API Endpoints

- `/summarize/single` - Single-agent summarization
- `/summarize/multi` - Multi-agent summarization

Both endpoints accept POST requests with JSON body: `{"text": "Your text here..."}`

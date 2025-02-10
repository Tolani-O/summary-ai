FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building pymupdf
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    swig \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

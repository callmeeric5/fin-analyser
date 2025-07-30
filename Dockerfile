FROM python:3.10-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (standalone binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY .env . 
# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv (in-place editable mode)
RUN uv sync --locked

# Copy the rest of the app
COPY . .

# Streamlit env config
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/app.py"]

# Standard Python 3.11 image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies into a virtual environment
RUN uv sync --no-editable

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose the port (HF Spaces uses 7860 by default, but OpenEnv spec says 8000)
# We will use 8000 as per openenv.yaml and mapping in Space settings
EXPOSE 8000

# Healthcheck to verify the server is up
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

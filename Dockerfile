# Use Python 3.12 slim base image
FROM python:3.12-slim

# Install system dependencies (ffmpeg is required for audio format conversion)
# libgomp1 is required by ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files for caching dependency layer
COPY pyproject.toml uv.lock* README.md /app/

# Install dependencies (without installing the project package itself)
RUN uv sync --frozen --no-install-project --no-dev

# Copy project source, configuration, and entrypoint script
COPY src /app/src
COPY voices_config.json /app/
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Install the project package
RUN uv sync --frozen --no-dev

# Pre-download the default voice model during the image build stage
RUN mkdir -p /app/default_voices && \
    VOICES_DIR=/app/default_voices uv run python -m kb_tts.download

# Set runtime environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV VOICES_DIR=/data/voices
ENV TTS_CONFIG_PATH=/app/voices_config.json

# Create volume mount point for persistent voice cache
RUN mkdir -p /data/voices
VOLUME ["/data/voices"]

# Expose server port
EXPOSE 8000

# Set entrypoint to initialize templates inside volume on startup
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the application
CMD ["uv", "run", "python", "-m", "kb_tts"]

# NBA Stat Predictor Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p data models logs plots backups

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory and set permissions
RUN chmod +x setup.py main.py && \
    chown -R nobody:nogroup /app

# Switch to non-root user
USER nobody

# Set environment variables
ENV PYTHONPATH=/app/src
ENV NBA_PREDICTOR_DB_PATH=/app/data/nba_data.db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sqlite3; conn = sqlite3.connect('data/nba_data.db'); conn.close()" || exit 1

# Default command
CMD ["python", "main.py", "predict"]

# Labels
LABEL maintainer="NBA Stat Predictor"
LABEL description="Machine learning system for NBA player stat prediction"
LABEL version="1.0" 
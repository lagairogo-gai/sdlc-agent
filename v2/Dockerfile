# Dockerfile - Simplified Requirements Agent
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete agent implementation
COPY requirements_agent_complete.py .

# Create directories
RUN mkdir -p /app/logs /app/data

# Create non-root user for security
RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "requirements_agent_complete.py"]
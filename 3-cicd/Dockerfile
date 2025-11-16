FROM python:3.12-slim

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y -qq unzip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY best_config.json .

# Copy schema and run_id if they exist
COPY training_schema.json* ./
COPY run_id.txt* ./

# Copy model artifacts (from GitHub Actions artifact)
COPY models/ ./models/

# Verify model exists
RUN if [ -d models/model ]; then \
      echo "✅ Model found at /app/models/model"; \
      ls -la models/model; \
    else \
      echo "⚠️  Model directory not found at /app/models/model"; \
      echo "Available files in /app:"; \
      ls -lR /app; \
    fi

# Set environment variables for the app
ENV MODEL_URI=./models/model
ENV SCHEMA_PATH=./training_schema.json
ENV PORT=9696
ENV ALLOW_DUMMY=0

# Expose the port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:9696/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
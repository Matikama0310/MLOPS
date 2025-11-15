FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# deps
COPY 3-cicd/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt && pip install "uvicorn[standard]" mlflow

# app + local artifacts (copy your exported MLflow model dir to ./model)
COPY 3-cicd/app.py ./app.py
COPY 3-cicd/training_schema.json ./training_schema.json
COPY 3-cicd/model ./model

ENV MODEL_URI=./model SCHEMA_PATH=./training_schema.json
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=2s --retries=12 CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
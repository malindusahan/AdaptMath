# Production Multi-Stage Dockerfile for Student Personalization Memory Service
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# The pinned psycopg binary wheel does not require compiler or libpq headers.
# Keep the image independent of Debian package mirrors; Python's standard
# library is sufficient for the container health check below.

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PYTHONPATH=/app

# Copy application source code and models
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).close()"]

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

COPY requirements-test-container.txt .
RUN pip install -r requirements-test-container.txt
COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

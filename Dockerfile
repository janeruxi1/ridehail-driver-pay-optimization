# Quote service image: the ridepay package with API extras and the model artifact. No data, no Spark.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"
COPY models ./models

# run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

ENV RIDEPAY_MODEL=/app/models/quote_model.json
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ridepay.api:app", "--host", "0.0.0.0", "--port", "8000"]

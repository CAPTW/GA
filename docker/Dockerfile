FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app/services

WORKDIR /app

COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY services ./services
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && pip install -e .[ops,tracking]

CMD ["python", "-m", "uvicorn", "ga_ops.app:app", "--host", "0.0.0.0", "--port", "8000"]

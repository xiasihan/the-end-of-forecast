FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY endforecast/ ./endforecast/

RUN pip install --no-cache-dir -e . fastapi uvicorn python-multipart sse-starlette

EXPOSE 8000

CMD ["uvicorn", "endforecast.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
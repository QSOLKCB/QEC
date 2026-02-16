FROM python:3.11-slim

WORKDIR /app

COPY dev-requirements.txt .

RUN pip install --no-cache-dir -r dev-requirements.txt

COPY src/ src/
COPY tests/ tests/

ENV PYTHONPATH=src

CMD ["python", "-m", "pytest", "-v"]

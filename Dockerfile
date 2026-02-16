FROM python:3.11-slim

WORKDIR /app

# Install dev dependencies
COPY dev-requirements.txt .
RUN pip install --no-cache-dir -r dev-requirements.txt

# Copy source + tests
COPY src/ src/
COPY tests/ tests/

ENV PYTHONPATH=src

CMD ["python", "-m", "pytest", "-v"]

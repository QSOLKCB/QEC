FROM python:3.11-slim

# Existing repository-bound regression assertions require Git and full history.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir -e '.[dev]'
CMD ["python", "-m", "pytest", "-q"]

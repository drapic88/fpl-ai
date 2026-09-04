FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FPL_CACHE_DIR=/app/.cache

WORKDIR /app

# Install dependencies in a separate layer for build cache optimization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure cache directory exists
RUN mkdir -p /app/.cache

ENTRYPOINT ["python", "-m", "fplai"]
CMD ["--help"]

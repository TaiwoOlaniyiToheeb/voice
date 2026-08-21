# Dockerfile for the Voice/Text Command Classifier API backend.
# Using Docker (rather than a buildpack) avoids the "PocketSphinx needs
# system libraries" problem some platforms hit with plain Python
# buildpacks -- we install what PocketSphinx needs explicitly below.

FROM python:3.11-slim

# System dependencies PocketSphinx needs to build/run correctly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway/Fly.io all inject a PORT environment variable at runtime;
# default to 8000 for local `docker run` testing.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

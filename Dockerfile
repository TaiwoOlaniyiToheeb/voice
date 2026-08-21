FROM python:3.12.14 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# PortAudio dev headers + build tools needed to compile pyaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.14-slim
WORKDIR /app

# PortAudio runtime lib needed for pyaudio to import at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv .venv/
COPY . .

EXPOSE 8080
CMD ["/app/.venv/bin/streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
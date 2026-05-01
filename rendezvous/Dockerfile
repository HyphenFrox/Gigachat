# Minimal container for GCP Cloud Run.
#
# Cloud Run honours the PORT env var (defaults to 8080) and routes
# inbound HTTPS traffic to it. We don't expose any ports in this
# Dockerfile because Cloud Run sets the listen port via env, not
# via Docker's EXPOSE.
#
# Build:    docker build -t gigachat-rendezvous .
# Run:      docker run -p 8080:8080 gigachat-rendezvous
# Deploy:   gcloud run deploy gigachat-rendezvous --source .

FROM python:3.13-slim

# Reduce image size + speed up cold-starts: -slim tags don't pull in
# the test suite or full GCC. The cryptography wheel is pre-built for
# musl-friendly slim variants so we don't need a build chain here.

WORKDIR /app

# Install deps in a separate layer so changes to main.py don't bust
# the cache. Pinned versions match `requirements.txt` exactly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Cloud Run sets PORT — uvicorn picks it up from the env var inside
# main.py's __main__ block. Direct CMD entry is faster on cold-start
# than going through `python main.py` because it skips the import
# graph traversal.
ENV PYTHONUNBUFFERED=1
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

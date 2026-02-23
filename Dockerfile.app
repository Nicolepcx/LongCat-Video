# ============================================================================
# LongCat-Video Avatar – Gradio UI  (lightweight, no GPU)
# ============================================================================
# Deploy on DigitalOcean App Platform, Railway, Fly.io, etc.
#
# Required env vars:
#   RUNPOD_API_KEY       – your RunPod API key
#   RUNPOD_ENDPOINT_ID   – your serverless endpoint ID
#   GRADIO_USERNAME      – login username  (default: admin)
#   GRADIO_PASSWORD      – login password  (REQUIRED for protection)
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]

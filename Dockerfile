FROM python:3.11-slim

# System deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Hugging Face Spaces requires the app to run as a non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user . /app

USER user

# Hugging Face Spaces routes traffic to port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "src/app.py", "--host", "0.0.0.0", "--port", "7860"]

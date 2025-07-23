FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

LABEL maintainer="Pascal Debus" \
      description="QML Playground" \
      version="1.0"

# Set working directory inside the container
WORKDIR /app

# Install curl and dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install (skip torch)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir $(grep -v "torch" /app/requirements.txt)

# Copy the app code from ./app in host into /app in container
COPY app/ /app/

# Create and use non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8050/ || exit 1

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8050

# Start the Dash app using gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050"]

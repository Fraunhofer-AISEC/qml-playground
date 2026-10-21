FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

LABEL maintainer="Pascal Debus" \
      description="QML Playground" \
      version="1.0" \
      org.opencontainers.image.source="https://github.com/fraunhofer-aisec/qml-playground"

# Set working directory inside the container
WORKDIR /app

# Install curl and dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (skip torch as it's in base image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(grep -v "torch" /app/requirements.txt)

# Copy the app code
COPY app/ /app/

# Create and use non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

# Healthcheck - Updated to use the correct path
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8050/qml-playground/ || exit 1

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8050

# Start the Dash app using gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120"]

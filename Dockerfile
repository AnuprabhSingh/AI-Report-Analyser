# Multi-stage Dockerfile for Medical Interpreter Application
# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend-react/package.json frontend-react/package-lock.json* ./

# Install dependencies
RUN npm ci --only=production || npm install

# Copy frontend source
COPY frontend-react/ ./

# Build frontend
RUN npm run build

# Stage 2: Python Backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    ghostscript \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY src/ ./src/
COPY templates/ ./templates/
COPY main.py .
COPY verify_setup.py .

# Copy ML models if they exist
COPY models/ ./models/

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/dist ./static/

# Create data directories
RUN mkdir -p data/processed data/sample_reports outputs

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/api.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "src.api:app"]

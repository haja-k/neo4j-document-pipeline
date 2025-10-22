# Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install deps
RUN pip install --no-cache-dir --upgrade pip

# Install python-multipart explicitly
RUN pip install --no-cache-dir python-multipart

# Copy requirements first (better build caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your app source
COPY . /app

# Expose the FastAPI port
EXPOSE 8686

# Start FastAPI automatically
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8686"]

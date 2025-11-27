# Use official Playwright Python image (has Playwright preinstalled)
FROM mcr.microsoft.com/playwright/python:v1.47.0-focal

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔴 IMPORTANT: install browser binaries inside the image
RUN playwright install --with-deps chromium

# Copy rest of the code
COPY . .

# Environment variables will be set on the platform
ENV PORT=8000

# Expose port (Railway will set actual port via $PORT)
EXPOSE 8000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

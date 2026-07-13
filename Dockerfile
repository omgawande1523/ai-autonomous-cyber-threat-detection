FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libpcap-dev \
    tshark \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all codebase files
COPY . /app/

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Default command to run (can be overridden in docker-compose)
CMD ["streamlit", "run", "backend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Step 1: Use lightweight Python image
FROM python:3.10-slim

# Step 2: Set work directory inside container
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy only requirements first (for caching)
COPY requirements.txt .

# Step 5: Install dependencies (CPU-only PyTorch to save space)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Step 6: Copy your code
COPY . .

# Step 7: Expose FastAPI port
EXPOSE 8000

# Step 8: Start FastAPI (which also starts Telegram bot)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

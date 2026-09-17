FROM python:3.11-slim

WORKDIR /app

# 1. Install CPU-only PyTorch FIRST so it doesn't download Nvidia CUDA drivers!
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Now copy and install the rest of your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your code and set the Python path
COPY . .
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
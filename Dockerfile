FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# mlx는 Apple Silicon 전용이므로 컨테이너에 설치하지 않는다
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs models data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "main.py"]

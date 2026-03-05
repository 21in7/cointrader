FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY log_parser.py .
COPY dashboard_api.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
EXPOSE 8080
CMD ["./entrypoint.sh"]

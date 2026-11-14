FROM python:3.12-slim
WORKDIR /app
COPY web ./web
EXPOSE 8787
ENV PORT=8787
CMD ["python", "web/app.py"]

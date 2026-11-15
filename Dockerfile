FROM python:3.12-slim
WORKDIR /app
COPY web ./web
COPY dryrun ./dryrun
COPY data ./data
COPY prompts ./prompts
COPY status ./status
COPY CODEX.md SUBMISSION.md MORNING.md ./
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir 'psycopg[binary]>=3.2'
EXPOSE 8787
ENV PORT=8787
CMD ["python", "web/app.py"]

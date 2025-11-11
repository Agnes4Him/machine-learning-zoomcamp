FROM python:3.12.3-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

COPY [".python-version", "pyproject.toml", "uv.lock", "./"]

ENV PATH="/code/.venv/bin:$PATH"

RUN uv sync --locked

COPY ["predict.py", "model.bin", "./"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

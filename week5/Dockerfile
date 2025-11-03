FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

COPY [".python-version", "pyproject.toml", "uv.lock", "./"]

ENV PATH="/code/.venv/bin:$PATH"

# Install packages system wide
RUN uv sync --locked

COPY "predict.py" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696", "--reload"]
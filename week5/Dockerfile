FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app

COPY ["pyproject.toml", "uv.lock", "./"]

# Install packeages system wide
RUN uv add ...

COPY ./ ./

EXPOSE 9696

CMD [ "uvicorn", "predict:app", "--host 0.0.0.0", "--port 9696", "--reload" ]
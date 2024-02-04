FROM python:3.9-slim
RUN apt-get update && apt-get install -y gcc
RUN apt install -y libpq-dev
RUN pip install poetry

WORKDIR /app
COPY ./ ./
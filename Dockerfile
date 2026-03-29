FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir duckdb==1.5.1 && \
    python -c "import duckdb; duckdb.connect(':memory:').execute('INSTALL spatial; LOAD spatial;')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py shadows.py osm_data.py weather_data.py index.html ./

ARG DB_URL=https://github.com/JuliusBec/munich-sunshine/releases/download/v1.0/munich.duckdb
RUN wget -q -O munich.duckdb "$DB_URL"

EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

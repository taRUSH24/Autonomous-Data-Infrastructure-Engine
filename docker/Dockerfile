FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ADIE_DUCKDB_PATH=/app/data/warehouse/adie.duckdb \
    DBT_PROFILES_DIR=/app/docker/dbt_profiles \
    DAGSTER_HOME=/app/.dagster_home \
    PYTHONPATH=/app

RUN mkdir -p /app/data/warehouse /app/.dagster_home "$DBT_PROFILES_DIR"

RUN dbt deps --project-dir dbt_project --profiles-dir "$DBT_PROFILES_DIR" || true

EXPOSE 3000

CMD ["dagster", "dev", "-m", "dagster_project.definitions", "-h", "0.0.0.0", "-p", "3000"]

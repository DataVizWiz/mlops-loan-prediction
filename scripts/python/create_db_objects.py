"""Create database for Model tracking."""

from os import environ
from backend.utilities.sql import PgSql

if __name__ == "__main__":
    un = environ["DB_USER"]
    pw = environ["DB_PASSWORD"]

    pg = PgSql("postgres", un, pw, "localhost", 5432)
    pg.create_database("logs")

    pg = PgSql("logs", un, pw, "localhost", 5432)
    query = """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id SERIAL PRIMARY KEY
        ,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ,model VARCHAR(20) NOT NULL UNIQUE
        ,r2_score FLOAT
    );
    """
    pg.execute_query(query)
    pg.insert_row(table="model_metrics", model="my_model_name", r2_score=0.92)

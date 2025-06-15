"""SQL classes."""

import psycopg2
from typing import Dict, List, Tuple, Union


class PgSql:
    """Apply methods to a PG database."""

    def __init__(self, database: str, user: str, password: str, host: str, port: str):
        """Init"""
        self.conn = psycopg2.connect(
            database=database, user=user, host=host, password=password, port=port
        )

    def create_database(self, database: str):
        """Create a database."""
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            check = f"SELECT 1 FROM pg_database WHERE datname = '{database}';"
            cur.execute(check)
            exists = cur.fetchone()

            if exists:
                msg = f"Database {database} already exists."
                msg += " No further action required."
                print(msg)
            else:
                cur.execute(f"CREATE DATABASE {database};")

    def execute_query(self, query: str) -> Union[List[Tuple], None]:
        """Execute a query and return results."""
        with self.conn.cursor() as cur:
            cur.execute(query)

            try:
                rows = cur.fetchall()
                self.conn.commit()
                print("Query executed successfully with rows.")
                return rows
            except psycopg2.ProgrammingError as e:
                if "no results to fetch" in str(e):
                    self.conn.commit()
                    print("Query executed successfully without rows.")

    def insert_row(self, table: str, schema: str = "public", **kwargs: Dict[str, str]):
        """Insert a row into a table."""
        with self.conn.cursor() as cur:
            fields = "(" + ", ".join(tuple(kwargs.keys())) + ")"
            placeholders = ", ".join(["%s"] * len(kwargs))
            values = tuple(kwargs.values())

            query = f"INSERT INTO {schema}.{table} {fields} VALUES({placeholders});"
            cur.execute(query, values)
            self.conn.commit()
        print(f"Row inserted into {schema}.{table}.")

import os
from dotenv import load_dotenv
import psycopg2
import sys

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/.env"))
print(f"Running: {__file__}", flush=True)

class PostgresDB:
    @staticmethod
    def connect():
        try:
            connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                dbname=os.getenv("POSTGRES_DB_NAME"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                connect_timeout=5
            )
            return connection
        except Exception as e:
            print(f"Connection error: {e}", file=sys.stderr, flush=True)
            return None 
        
    @staticmethod
    def close(connection):
        try:
            if connection is not None:
                connection.close()
                print("Connection closed.", flush=True)
        except Exception as e:
            print(f"Error closing connection: {e}", file=sys.stderr, flush=True)

    @staticmethod
    def fetch_one():
        
        connection = None
        cursor = None
        
        try: 
            connection = PostgresDB.connect()
            if connection is None:
                return None
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM public.raw_data LIMIT 1;")
            row = cursor.fetchone()
            return row
        
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return None
        
        finally:
            if cursor is not None:
                try:cursor.close()
                except Exception: pass
            if connection is not None:
                try: PostgresDB.close(connection)
                except Exception: pass
    @staticmethod
    def fetch_all_data():
        
        connection = None
        cursor = None
        
        try: 
            connection = PostgresDB.connect()
            if connection is None:
                return None
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM public.raw_data;")
            rows = cursor.fetchall()
            return rows
        
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return None
        
        finally:
            if cursor is not None:
                try: cursor.close()
                except Exception: pass
            if connection is not None:
                try: PostgresDB.close(connection)
                except Exception: pass
                
    @staticmethod
    def get_columns():
        connection = None
        cursor = None
        try:
            connection = PostgresDB.connect()
            if connection is None:
                return None
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM public.raw_data LIMIT 0;")
            cols = [d.name if hasattr(d, "name") else d[0] for d in cursor.description]
            return cols
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return None
        finally:
            if cursor is not None:
                try: cursor.close()
                except Exception: pass
            if connection is not None:
                try: PostgresDB.close(connection)
                except Exception: pass

    @staticmethod
    def fetch_since(ts, ts_col="timestamp"):
        """
        Fetch rows newer than the given timestamp from public.raw_data.
        ts: a Python datetime (aware or naive), or string compatible with psycopg2.
        ts_col: the name of the timestamp column (default 'timestamp').
        """
        connection = None
        cursor = None
        try:
            connection = PostgresDB.connect()
            if connection is None:
                return None
            cursor = connection.cursor()
            # quote identifier safely by using double quotes; assumes ts_col is valid
            query = f'SELECT * FROM public.raw_data WHERE "{ts_col}" > %s ORDER BY "{ts_col}" ASC;'
            cursor.execute(query, (ts,))
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return None
        finally:
            if cursor is not None:
                try: cursor.close()
                except Exception: pass
            if connection is not None:
                try: PostgresDB.close(connection)
                except Exception: pass

import os
from dotenv import load_dotenv
import psycopg2
import sys

#import file from Mee-Tiger directory

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "/Users/peakypooka/Library/VSCode_Backup/Dev_Team/Mee-Tiger/.env"))
print(f"Running: {__file__}", flush=True)

def fetch_all_data():
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB_NAME"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            connect_timeout=5
        )
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM public.raw_data LIMIT 5;")
        rows = cursor.fetchall()
        return rows
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

try:
    print(f"Connecting to {os.getenv('POSTGRES_HOST')} ...", flush=True)
    connection = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB_NAME"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=5
    )
    print("Connected.", flush=True)

    current_session = connection.cursor()
    current_session.execute("SELECT current_database(), current_user;")
    row = current_session.fetchone()
    print(f"DB: {row[0]} | USER: {row[1]}", flush=True)
    
    data = fetch_all_data()
    for data in data:
        print(data, flush=True)
    current_session.close()
    

# Close the connection    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
finally:
    try:
        if 'connection' in locals() and connection is not None:
            connection.close()
            print("Connection closed.", flush=True)
    except Exception as e:
        print(f"ERROR during close: {e}", file=sys.stderr, flush=True)

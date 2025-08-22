import psycopg2
import sys

print(f"Running: {__file__}", flush=True)

host="192.168.15.148"

try:
    print(f"Connecting to {host} ...", flush=True)
    connection = psycopg2.connect(
        host=host,
        port="5432",
        dbname="my_new_database",
        user="postgres",
        password="postgres",
        connect_timeout=5
    )
    print("Connected.", flush=True)

    current_session = connection.cursor()
    current_session.execute("SELECT current_database(), current_user;")
    row = current_session.fetchone()
    print(f"DB: {row[0]} | USER: {row[1]}", flush=True)
    
    current_session.execute("SELECT * FROM employees;")
    
    #Print all Contents inside the Table
    rows = current_session.fetchall()
    for rows in rows:
        print(rows, flush=True)
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


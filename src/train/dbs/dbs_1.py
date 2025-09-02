import pandas as pd

from pathlib import Path
from src.db.postgres import PostgresDB

CSV_PATH = Path(__file__).parent / "exported_data.csv"

class DBS_Data:
    
    def load_all_data():
        data = PostgresDB.fetch_all_data()
        return data
    
    def load_single_row():
        data = PostgresDB.fetch_one()
        return data

# Test print single row
print("Testing DBS_Data.load_single_row()...", flush=True)
row = DBS_Data.load_single_row()
print(f"Fetched row: {row}", flush=True)
if row is None:
    print("No data fetched.", flush=True)
else:
    print(f"Fetched {len(row)} columns.", flush=True)

# Save to CSV and append only new data (query only the delta)
data = None
last_ts = None

# Determine columns from DB
cols = PostgresDB.get_columns()

# Find watermark from existing CSV
if CSV_PATH.exists():
    try:
        # If we know the timestamp column name, we can parse by that; otherwise parse first column
        if cols and len(cols) > 0:
            ts_col_name = cols[0]
            existing_df = pd.read_csv(CSV_PATH, parse_dates=[ts_col_name])
            last_ts = existing_df[ts_col_name].max() if not existing_df.empty else None
        else:
            existing_df = pd.read_csv(CSV_PATH, parse_dates=[0])
            last_ts = existing_df.iloc[:, 0].max() if not existing_df.empty else None
    except Exception as e:
        print(f"Warning: failed to read existing CSV ({e}), falling back to full fetch.", flush=True)
        last_ts = None

# Fetch only new rows if we have a watermark, else fetch all
if last_ts is not None and not pd.isna(last_ts):
    # Use first column name as timestamp column unless you want to hardcode it
    ts_col_name = cols[0] if cols else "timestamp"
    print(f"Fetching rows since {last_ts} using ts_col '{ts_col_name}' ...", flush=True)
    data = PostgresDB.fetch_since(last_ts, ts_col=ts_col_name)
else:
    print("No watermark found; fetching full dataset once.", flush=True)
    data = PostgresDB.fetch_all_data()

if data is None or len(data) == 0:
    print("No data fetched for CSV export.", flush=True)
else:
    # Build DataFrame with column names if available
    df = pd.DataFrame(data, columns=cols if cols else None)

    # Ensure timestamp column is parsed as datetime
    if cols and len(cols) > 0:
        ts_col_name = cols[0]
        try:
            df[ts_col_name] = pd.to_datetime(df[ts_col_name], errors='coerce', utc=True)
        except Exception:
            pass
    else:
        # if no cols available, try first column
        try:
            df[0] = pd.to_datetime(df[0], errors='coerce', utc=True)
        except Exception:
            pass

    # Ensure directory exists
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append-only write (header only if file doesn't exist or empty)
    write_header = not CSV_PATH.exists()
    if not write_header:
        try:
            if CSV_PATH.stat().st_size == 0:
                write_header = True
        except Exception:
            pass

    df.to_csv(CSV_PATH, mode="a" if CSV_PATH.exists() else "w", header=write_header, index=False)
    print(f"Appended {len(df)} rows to {CSV_PATH}", flush=True)

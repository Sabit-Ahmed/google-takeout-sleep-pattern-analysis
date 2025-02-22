import sqlite3
from os.path import expanduser
import pandas as pd

def save_to_csv_pandas(data, filename="data/screen_time_data.csv"):
    headers = ["app", "usage", "start_time", "end_time", "created_at", "tz", "device_id", "device_model"]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")

def query_database():
    # Connect to the SQLite database
    knowledge_db = expanduser("~/Library/Application Support/Knowledge/knowledgeC.db")
    with sqlite3.connect(knowledge_db) as con:
        cur = con.cursor()

        # Execute the SQL query to fetch data
        # Modified from https://rud.is/b/2019/10/28/spelunking-macos-screentime-app-usage-with-r/
        query = """
        SELECT
            ZOBJECT.ZVALUESTRING AS "app", 
            (ZOBJECT.ZENDDATE - ZOBJECT.ZSTARTDATE) AS "usage",
            (ZOBJECT.ZSTARTDATE + 978307200) as "start_time", 
            (ZOBJECT.ZENDDATE + 978307200) as "end_time",
            (ZOBJECT.ZCREATIONDATE + 978307200) as "created_at", 
            ZOBJECT.ZSECONDSFROMGMT AS "tz",
            ZSOURCE.ZDEVICEID AS "device_id",
            ZMODEL AS "device_model"
        FROM
            ZOBJECT 
            LEFT JOIN
            ZSTRUCTUREDMETADATA 
            ON ZOBJECT.ZSTRUCTUREDMETADATA = ZSTRUCTUREDMETADATA.Z_PK 
            LEFT JOIN
            ZSOURCE 
            ON ZOBJECT.ZSOURCE = ZSOURCE.Z_PK 
            LEFT JOIN
            ZSYNCPEER
            ON ZSOURCE.ZDEVICEID = ZSYNCPEER.ZDEVICEID
        WHERE
            ZSTREAMNAME = "/app/usage"
        ORDER BY
            ZSTARTDATE DESC
        """
        cur.execute(query)

        # Fetch all rows from the result set
        return cur.fetchall()

data = query_database()
# Run the pandas-based function
save_to_csv_pandas(data)
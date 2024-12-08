import sqlite3
import pandas as pd # type: ignore
import json


if __name__ == "__main__":
  df = pd.read_csv("obj_analysis_init_data.csv")

  # Connect to SQLite database (or create it if it doesn't exist)
  conn = sqlite3.connect('objectives.db')
  cursor = conn.cursor()

  # Create a table based on the dataframe columns (adjust column types if necessary)
  columns = df.columns
  columns = f"id INTEGER PRIMARY KEY AUTOINCREMENT, {columns[0]} TEXT NOT NULL, {columns[1]} INTEGER NOT NULL, {columns[2]} INTEGER NOT NULL, {columns[3]} INTEGER NOT NULL, {columns[4]} INTEGER NOT NULL, {columns[5]} TEXT NOT NULL"
  create_table_query = f"CREATE TABLE IF NOT EXISTS objectives ({columns})"
  cursor.execute(create_table_query)

  # Insert filtered data into the table
  for row in df.itertuples(index=False, name=None):
    in_obj = json.dumps(row[0].split())
    new_obj = json.dumps(row[5].split())
    insert_query = f"INSERT INTO objectives ({', '.join([col for col in df.columns])}) VALUES (?, ?, ?, ?, ?, ?)"
    cursor.execute(insert_query, (in_obj, row[1], row[2], row[3], row[4], new_obj))

  # Commit the transaction and close the connection
  conn.commit()
  conn.close()
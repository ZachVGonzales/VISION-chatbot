import sqlite3
import pandas as pd # type: ignore
import json
import argparse


def init_params():
  parser = argparse.ArgumentParser(prog="init_obj_db.py", 
                                   description="init the training database for objectives")
  parser.add_argument("db_path", help="path to the training database")
  parser.add_argument("data_path", help="path to csv data file")
  return parser.parse_args()


if __name__ == "__main__":
  params = init_params()
  db_path = params.db_path
  data_path = params.data_path
  df = pd.read_csv(data_path)

  # Connect to SQLite database (or create it if it doesn't exist)
  conn = sqlite3.connect(db_path)
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
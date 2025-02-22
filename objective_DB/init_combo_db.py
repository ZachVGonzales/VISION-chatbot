import sqlite3
import pandas as pd # type: ignore
import json
import argparse


CSV_COLUMNS = ["input_objective","audience_score","behavior_score","condition_score","degree_score","vision_type1","vision_type2","BT1","BT2","BT3","BT4","BT5","BT6","new_objective"]
DB_COLUMNS = ["input_objective","abcd","type","blooms","new_objective"]


if __name__ == "__main__":
  db_path = "objectives_combo.db"
  data_path = "objectives_combo.csv"
  df = pd.read_csv(data_path)

  # Connect to SQLite database (or create it if it doesn't exist)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create a table based on the dataframe columns (adjust column types if necessary)
  #columns = df.columns
  columns = f"id INTEGER PRIMARY KEY AUTOINCREMENT, {DB_COLUMNS[0]} TEXT NOT NULL, {DB_COLUMNS[1]} TEXT NOT NULL, {DB_COLUMNS[2]} TEXT NOT NULL, {DB_COLUMNS[3]} TEXT NOT NULL, {DB_COLUMNS[4]} TEXT NOT NULL"
  create_table_query = f"CREATE TABLE IF NOT EXISTS objectives ({columns})"
  cursor.execute(create_table_query)

  # Insert filtered data into the table
  print(df)
  for row in df.itertuples(index=False, name=None):
    in_obj = json.dumps(row[0].split())
    new_obj = json.dumps(row[13].split())
    abcd_labels = json.dumps([row[1], row[2], row[3], row[4]])
    type_labels = json.dumps([(row[5]), row[6]])
    blooms_labels = json.dumps([row[7], row[8], row[9], row[10], row[11], row[12]])
    insert_query = f"INSERT INTO objectives ({', '.join([col for col in DB_COLUMNS])}) VALUES (?, ?, ?, ?, ?)"
    cursor.execute(insert_query, (in_obj, abcd_labels, type_labels, blooms_labels, new_obj))

  # Commit the transaction and close the connection
  conn.commit()
  conn.close()
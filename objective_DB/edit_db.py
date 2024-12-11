import sqlite3
import json
import argparse
import random


CONDITION_PRES = ["given this", "without", "with", "Consulting this", "lacking"]


def init_params():
  parser = argparse.ArgumentParser(prog="init_obj_db.py", 
                                   description="init the training database for objectives")
  parser.add_argument("db_path", help="path to csv data file")
  return parser.parse_args()


if __name__ == "__main__":
  params = init_params()
  db_path = params.db_path

  # Connect to SQLite database (or create it if it doesn't exist)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # get all rows:
  cursor.execute("SELECT * FROM objectives")
  rows = cursor.fetchall()

  # Insert filtered data into the table
  for i, row in enumerate(rows):
    if ((i - 4) % 20) == 0:
      obj = " ".join(json.loads(row[1]))
      new_obj = f"{random.choice(CONDITION_PRES)} {obj}".split()
      cursor.execute("UPDATE objectives SET input_objective = ? WHERE id = ?", (json.dumps(new_obj), i+1))

  # Commit the transaction and close the connection
  conn.commit()
  conn.close()
import sqlite3
import json
import torch
from torch.utils.data import Dataset
import sys


class SQLiteDataset(Dataset):
  def __init__(self, db_connection: sqlite3.Connection, table_name: str, load_columns: list[str] | None) -> None:
    super().__init__()
    self.conn = db_connection
    self.cursor = self.conn.cursor()
    self.table_name = table_name
    self.load_columns = load_columns

    # find the num rows in database
    self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
    self.total_rows = self.cursor.fetchone()[0]

  def __len__(self):
    return self.total_rows

  def __getitem__(self, index):
    self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1 OFFSET {index}")
    column_names = [description[0] for description in self.cursor.description]
    cname2idx = {n:i for i, n in enumerate(column_names)}
    example = self.cursor.fetchone()
    new_example = {}

    try:
      for column_name in column_names:
        if self.load_columns and column_name in self.load_columns:
            new_example[column_name] = json.loads(example[cname2idx[column_name]])
        else:
          new_example[column_name] = example[cname2idx[column_name]]
    except Exception as e:
      print(f"exception {e}", file=sys.stderr)
      print(f"index: {index}", file=sys.stderr)
      print(f"table: {self.table_name}", file=sys.stderr)
      print(f"example: {example}", file=sys.stderr)
      return None

    return new_example


class SQLiteDatasetEncoded(Dataset):
  def __init__(self, db_connection: sqlite3.Connection, table_name: str, load_columns: list[str] | None, tokenizer, encode_fn, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
    super().__init__()
    self.conn = db_connection
    self.cursor = self.conn.cursor()
    self.table_name = table_name
    self.load_columns = load_columns
    self.tokenizer = tokenizer
    self.device = device
    self.encode_fn = encode_fn

    # find the num rows in database
    self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
    self.total_rows = self.cursor.fetchone()[0]
    
  
  def __len__(self):
    return self.total_rows

  def __getitem__(self, index):
    self.cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 1 OFFSET {index}")
    column_names = [description[0] for description in self.cursor.description]
    cname2idx = {n:i for i, n in enumerate(column_names)}
    example = self.cursor.fetchone()
    new_example = {}

    if example is None:
      return None

    try:
      for column_name in column_names:
        if self.load_columns and column_name in self.load_columns:
          new_example[column_name] = json.loads(example[cname2idx[column_name]])
        else:
          new_example[column_name] = example[cname2idx[column_name]]
    except Exception as e:
      print(f"exception {e}", file=sys.stderr)
      print(f"index: {index}", file=sys.stderr)
      print(f"table: {self.table_name}", file=sys.stderr)
      print(f"example: {example}", file=sys.stderr)
      return None

    return self.encode_fn(new_example, self.tokenizer, self.device)
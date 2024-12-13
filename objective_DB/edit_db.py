import sqlite3
import json
import argparse
import random
import os
from transformers import MarianMTModel, MarianTokenizer # type: ignore



CONDITION_PRES = ["given this", "without", "with", "Consulting this", "lacking"]



def init_params():
  parser = argparse.ArgumentParser(prog="edit_db.py", 
                                   description="init the training database for objectives")
  parser.add_argument("db_path", help="path to csv data file")
  parser.add_argument("null_exs", help="path to any null examples to be added")
  return parser.parse_args()



def translate(words: list[str], model, tokenizer) -> list[str]:
  text = " ".join(words)
  inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
  translated = model.generate(inputs, max_length=512)
  return tokenizer.decode(translated[0], skip_special_tokens=True).split()



if __name__ == "__main__":
  params = init_params()
  db_path = params.db_path
  null_exs = params.null_exs

  # init translation models
  model_name_forward0 = "Helsinki-NLP/opus-mt-en-es"
  tokenizer_forward0 = MarianTokenizer.from_pretrained(model_name_forward0)
  model_forward0 = MarianMTModel.from_pretrained(model_name_forward0)

  model_name_forward1 = "Helsinki-NLP/opus-mt-es-fr"
  tokenizer_forward1 = MarianTokenizer.from_pretrained(model_name_forward1)
  model_forward1 = MarianMTModel.from_pretrained(model_name_forward1)

  model_name_back = "Helsinki-NLP/opus-mt-fr-en"
  tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
  model_back = MarianMTModel.from_pretrained(model_name_back)

  # Connect to SQLite database (or create it if it doesn't exist)
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # get all rows:
  cursor.execute("SELECT * FROM objectives")
  rows = cursor.fetchall()

  # edit a small portion of the rows
  for i, row in enumerate(rows):
    if random.random() < 0.35:
      obj = json.loads(row[1])
      obj = translate(obj, model_forward0, tokenizer_forward0)
      obj = translate(obj, model_forward1, tokenizer_forward1)
      obj = translate(obj, model_back, tokenizer_back)
      cursor.execute("UPDATE objectives SET input_objective = ? WHERE id = ?", (json.dumps(obj), i+1))

    if random.random() < 0.5:
      obj = json.loads(row[6])
      obj = translate(obj, model_forward0, tokenizer_forward0)
      obj = translate(obj, model_forward1, tokenizer_forward1)
      obj = translate(obj, model_back, tokenizer_back)
      cursor.execute("UPDATE objectives SET new_objective = ? WHERE id = ?", (json.dumps(obj), i+1))

    print(f"{i}/{len(rows)} examples processed")
  conn.commit()

  # add some null examples
  if os.path.exists(null_exs):
    with open(null_exs, 'r') as null_file:
      for line in null_file:
        # grab a random target
        cursor.execute("SELECT new_objective FROM objectives ORDER BY RANDOM() LIMIT 1")
        target = json.loads(cursor.fetchone()[0])
        
        # insert the null value with random target
        cursor.execute("INSERT INTO objectives (input_objective, audience_score, behavior_score, condition_score, degree_score, new_objective) VALUES (?, ?, ?, ?, ?, ?)", 
                       (json.dumps(line.split()), 0, 0, 0, 0, json.dumps(target)))

  # Commit the transaction and close the connection
  conn.commit()
  conn.close()
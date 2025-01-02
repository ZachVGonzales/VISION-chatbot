from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import torch
import sqlite3
from utils.datasets import SQLiteDatasetEncoded
from utils.dataset_processing import encode_scoring, collate_scoring
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore
import numpy as np
import os
import argparse
import json



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4



def init_example_table(cursor: sqlite3.Cursor):
  # get the imperfect examples from the objectives table
  cursor.execute("SELECT input_objective, audience_score, behavior_score, condition_score, degree_score FROM objectives")
  rows = cursor.fetchall()
  for row in rows:
    input_obj, a, b ,c, d = row
    labels = [a, b, c, d]
    cursor.execute("INSERT INTO scoring_examples (text, labels) VALUES (?, ?)", (input_obj, json.dumps(labels)))
  
  # get the perfect examples from the objectives table
  cursor.execute("SELECT DISTINCT new_objective FROM objectives")
  rows = cursor.fetchall()
  for row in rows:
    obj = row[0]
    labels = [1, 1, 1, 1]
    cursor.execute("INSERT INTO scoring_examples (text, labels) VALUES (?, ?)", (obj, json.dumps(labels)))



def train_model(model, dataloaders: dict[str:DataLoader], optimizer, criterion, patience=5, n_epochs=20):
  best_loss = float('inf')
  epochs_no_improve = 0

  for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    for phase in ["train", "val"]:
      if phase == "train":
        model.train()
      else:
        model.eval()
      
      running_loss = 0.0
      preds, true_labels = [], []

      with tqdm(total=len(dataloaders[phase]), desc=f"{phase.capitalize()} Progress", unit="batch") as pbar:
        for batch in dataloaders[phase]:
          optimizer.zero_grad()
          batch = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch.items()}

          with torch.set_grad_enabled(phase == "train"):
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])
            preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            true_labels.extend(torch.sigmoid(logits).detach().cpu().numpy())

            if phase == "train":
              loss.backward()
              optimizer.step()
            
          running_loss += loss.item() * batch["input_ids"].size(0)
          pbar.set_postfix(loss=loss.item())
          pbar.update(1)
      
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      preds = np.array(preds)
      true_labels = np.array(true_labels)

      print(f"{phase} loss: {epoch_loss:.4f}")
      if phase == "val":
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          epochs_no_improve = 0
          torch.save(model.state_dict(), "./models/scoring.pt")
        else:
          epochs_no_improve += 1

        if epochs_no_improve == patience:
          print("EARLY STOPPING IMPLEMENTED")
          print("state dict saved at: ./models/scoring.pt")
          return
  
  print("DONE TRAINING")
  print("model dict saved at: ./models/scoring.pt")
  return



def init_params():
  parser = argparse.ArgumentParser(prog="train_scoring.py", 
                                   description="train the scoring model for the objective helper")
  parser.add_argument("db_path", help="path to the training database")
  parser.add_argument("model_path", help="path to saved model (None if no saved model exists)")
  return parser.parse_args()



if __name__ == "__main__":
  # init params
  params = init_params()
  db_path = params.db_path
  model_path = params.model_path

  # init the database connection
  connection = sqlite3.connect(db_path)
  cursor = connection.cursor()

  # init the total example table
  cursor.execute("CREATE TEMPORARY TABLE scoring_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, labels TEXT)")
  init_example_table(cursor)

  # split into train and test tables
  cursor.execute("SELECT COUNT(*) FROM scoring_examples")
  num_examples = cursor.fetchone()[0]
  cursor.execute("CREATE TEMPORARY TABLE train (id INTEGER PRIMARY KEY, text TEXT, labels TEXT)")
  cursor.execute("CREATE TEMPORARY TABLE test (id INTEGER PRIMARY KEY, text TEXT, labels TEXT)")
  cursor.execute(f"INSERT INTO test SELECT id, text, labels FROM scoring_examples ORDER BY RANDOM() LIMIT {int(num_examples * 0.1)}")
  cursor.execute("INSERT INTO train SELECT id, text, labels FROM scoring_examples WHERE id NOT IN (SELECT id FROM test)")
  cursor.execute("DROP TABLE scoring_examples")

  # init database interfaces for training
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  train_ds = SQLiteDatasetEncoded(connection, "train", ["text", "labels"], tokenizer, encode_scoring, DEVICE)
  test_ds = SQLiteDatasetEncoded(connection, "test", ["text", "labels"], tokenizer, encode_scoring, DEVICE)
  train_dl = DataLoader(dataset=train_ds, shuffle=True, collate_fn=collate_scoring, batch_size=BATCH_SIZE)
  test_dl = DataLoader(dataset=test_ds, shuffle=True, collate_fn=collate_scoring, batch_size=BATCH_SIZE)

  # init the model
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 4)
  if os.path.exists(model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
  model.to(DEVICE)

  # init other necessary training parameters
  criterion = torch.nn.BCEWithLogitsLoss()
  dataloaders = {"train": train_dl, "val": test_dl}
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  # now train on data
  train_model(model, dataloaders, optimizer, criterion)
  quit()
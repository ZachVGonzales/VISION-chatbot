from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import torch
import sqlite3
from utils.datasets import SQLiteDatasetEncoded
from utils.dataset_processing import encode_scoring, collate_scoring
from torch.utils.data import DataLoader
from utils.models import BERTMultiTaskClassifier
from tqdm import tqdm # type: ignore
import numpy as np
import os
import argparse
import json


"""
NOTE: model is trained to output it's prediction in a list (tensor) of float values.
The first 4 of these values represent how the model has scored the audience, behavior,
condition, and degree categories respectively. A high score means that the model is 
condfident that this category is present. The last 2 float values represent what 'type'
of objective the model thinks it is. This is shown by the following classsification:
last 2 values: ------------------ objective type:
    0, 0       ------------------ Cognitive Enabler
    0, 1       ------------------ Cognitive Terminal
    1, 0       ------------------ Performance Enabler
    1, 1       ------------------ Performance Terminal
In the case that the model does not recognize the input as an objective i.e. it scored
a very low score in all 4 of the first categories the output of the last 2 categories
should be ignored. This is because in the training the loss of these categories was 
ignored for examples where the first 4 categories were labeled as 0, 0, 0, 0. ( this
was done to avoid the model fitting to patterns that would worsen performance on actual
objectives).

EX: if the output of the model is: (1, 1, 1, 0, 0, 1)

Then this means that the model predicts there to be an audience, behavior, and condition
present in the objective; while the degree is not. The model also predicts that this 
objective is of type: Cognitive Terminal.
"""


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16



def init_example_table(cursor: sqlite3.Cursor):
  # get the examples from the objectives table
  cursor.execute("SELECT input_objective, abcd, type, blooms FROM objectives")
  rows = cursor.fetchall()
  for row in rows:
    input_obj, abcd, types, blooms = row
    abcd = json.loads(abcd)
    types = json.loads(types)
    blooms = json.loads(blooms)

    labels = abcd + types + blooms
    cursor.execute("INSERT INTO scoring_examples (text, labels) VALUES (?, ?)", (input_obj, json.dumps(labels)))


def train_model(model: BERTMultiTaskClassifier, dataloaders: dict[str:DataLoader], optimizer, patience=5, n_epochs=20):
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
            logits = model.forward(batch["input_ids"], attention_mask=batch["attention_mask"])
            labels = batch["labels"]
            labels = {"abcd": labels[:, :4], "types": labels[:, 4:6], "blooms": labels[:, 6:10]}
            loss = model.compute_loss(logits, labels)

            #preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            #true_labels.extend(torch.sigmoid(logits).detach().cpu().numpy())

            if phase == "train":
              loss.backward()
              optimizer.step()
            
          running_loss += loss.item() * batch["input_ids"].size(0)
          pbar.set_postfix(loss=loss.item())
          pbar.update(1)
      
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      #preds = np.array(preds)
      #true_labels = np.array(true_labels)

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


def custom_criterion(logits: torch.Tensor, labels: torch.Tensor):
  # Compute per-element loss
  loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')  
  loss = loss_fn(logits, labels)

  # True if first 4 labels are all zero
  mask_condition = (labels[:, :4].sum(dim=1) == 0)  
  mask = torch.ones_like(labels)

  # Zero out last 2 categories if condition met
  mask[mask_condition, 4:] = 0  
  masked_loss = loss * mask
  final_loss = masked_loss.sum() / mask.sum()

  return final_loss



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
  train_ds = SQLiteDatasetEncoded(db_connection=connection, 
                                  table_name="train", 
                                  load_columns=["text", "labels"], 
                                  tokenizer=tokenizer, 
                                  encode_fn=encode_scoring, 
                                  device=DEVICE)
  test_ds = SQLiteDatasetEncoded(db_connection=connection, 
                                 table_name="test", 
                                 load_columns=["text", "labels"], 
                                 tokenizer=tokenizer, 
                                 encode_fn=encode_scoring, 
                                 device=DEVICE)
  train_dl = DataLoader(dataset=train_ds, shuffle=True, collate_fn=collate_scoring, batch_size=BATCH_SIZE)
  test_dl = DataLoader(dataset=test_ds, shuffle=True, collate_fn=collate_scoring, batch_size=BATCH_SIZE)

  # init the model
  classes = {"abcd": (False, ["audience", "behavior", "condition", "degree"]), "types": (False, ["type1", "type2"]), "blooms": (True, ["BT1", "BT2", "BT3", "BT4", "BT5", "BT6"])}
  if os.path.exists(model_path):
    state_dict = torch.load(model_path)
  model = BERTMultiTaskClassifier(classes=classes, device=DEVICE)

  # init other necessary training parameters
  criterion = custom_criterion
  dataloaders = {"train": train_dl, "val": test_dl}
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  # now train on data
  train_model(model, dataloaders, optimizer)
  quit()
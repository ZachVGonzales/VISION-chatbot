from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification # type: ignore
from tqdm import tqdm # type: ignore
import torch
import sqlite3
import argparse
from utils.datasets import SQLiteDatasetEncoded
from utils.dataset_processing import encode_seq, collate_seq, score_seq, calc_reward, compute_loss
from torch.utils.data import DataLoader
import os



SEQ_MODEL_NAME = "google/t5-v1_1-base"
SCORE_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2



def init_example_table(cursor: sqlite3.Cursor):
  # get the imperfect examples from the objectives table
  cursor.execute("SELECT input_objective, new_objective FROM objectives")
  rows = cursor.fetchall()
  for row in rows:
    input_obj, new_obj = row
    cursor.execute("INSERT INTO seq_examples (input, target) VALUES (?, ?)", (input_obj, new_obj))

  # get the imperfect examples from the augmented table
  cursor.execute("SELECT input_objective, new_objective FROM augmented_objectives")
  rows = cursor.fetchall()
  for row in rows:
    input_obj, new_obj = row
    cursor.execute("INSERT INTO seq_examples (input, target) VALUES (?, ?)", (input_obj, new_obj))
  
  # get the perfect examples from the objectives table
  cursor.execute("SELECT DISTINCT new_objective FROM objectives")
  rows = cursor.fetchall()
  for row in rows:
    obj = row[0]
    cursor.execute("INSERT INTO seq_examples (input, target) VALUES (?, ?)", (obj, obj))



def train_seq2seq(seq_model, score_model, seq_tokenizer, score_tokenizer, dataloaders, criterion, optimizer: torch.optim.Optimizer, patience: int = 3, n_epochs: int = 20):
  best_loss = float('inf')
  epochs_no_improve = 0
  score_model.eval() # not training the score model

  # train seq_model for n number of epochs
  for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")

    # in each epoch train and evaluate model
    for phase in ["train", "val"]:
      if phase == "train":
        seq_model.train()
      else:
        seq_model.eval()
      
      # for calculating loss over the entire epoch
      running_loss = 0.0

      # Now loop through epoch a batch at a time (update progress bar along the way)
      with tqdm(total=len(dataloaders[phase]), desc=f"{phase.capitalize()} Progress", unit="batch") as pbar:
        for batch in dataloaders[phase]:
          optimizer.zero_grad()
          batch = {k: v.to(DEVICE) if type(v) is torch.Tensor else v for k, v in batch.items()}
          
          # get sequence prediction from seq2seq model
          with torch.set_grad_enabled(phase == "train"):
            outputs = seq_model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["target_ids"])
            logits = outputs.logits

            # get scores / rewards from score model
            generated_ids = seq_model.generate(batch["input_ids"])
            generated_texts = [seq_tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            scores = [score_seq(score_model, score_tokenizer, text, DEVICE) for text in generated_texts]
            rewards = [calc_reward(score) for score in scores]
            rewards = torch.tensor(rewards, device=DEVICE)

            # compute loss with criterion
            loss = criterion(logits, batch["target_ids"], rewards)

          # backpropogate only if training
          if phase == "train":
            loss.backward()
            optimizer.step()

          # add to running loss and step progress
          running_loss += loss.item() * batch["input_ids"].size(0)
          pbar.set_postfix(loss=loss.item())
          pbar.update(1)
      
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      print(f"{phase} loss: {epoch_loss:.4f}")

      # if validation check if best model and update patience
      if phase == "val":
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          epochs_no_improve = 0
          torch.save(seq_model.state_dict(), "./models/seq2seq.pt")
        else:
          epochs_no_improve += 1

        if epochs_no_improve == patience:
          print("EARLY STOPPING IMPLEMENTED")
          print("state dict saved at: ./models/seq2seq.pt")
          return
  
  print("DONE TRAINING")
  print("model dict saved at: ./models/seq2seq.pt")
  return



def init_params():
  parser = argparse.ArgumentParser(prog="train_scoring.py", 
                                   description="train the scoring model for the objective helper")
  parser.add_argument("db_path", help="path to the training database")
  parser.add_argument("score_path", help="path to saved scoring model (None if no saved model exists)")
  parser.add_argument("seq_path", help="path to the saved seq2seq model (None if no saved model exists)")
  return parser.parse_args()



if __name__ == "__main__":
  # init params
  params = init_params()
  db_path = params.db_path
  score_path = params.score_path
  seq_path = params.seq_path

  # init the database connection
  connection = sqlite3.connect(db_path)
  cursor = connection.cursor()

  # init the total example table
  cursor.execute("CREATE TEMPORARY TABLE seq_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, input TEXT, target TEXT)")
  init_example_table(cursor)

  # split into train and test tables
  cursor.execute("SELECT COUNT(*) FROM seq_examples")
  num_examples = cursor.fetchone()[0]
  cursor.execute("CREATE TEMPORARY TABLE train (id INTEGER PRIMARY KEY, input TEXT, target TEXT)")
  cursor.execute("CREATE TEMPORARY TABLE test (id INTEGER PRIMARY KEY, input TEXT, target TEXT)")
  cursor.execute(f"INSERT INTO test SELECT id, input, target FROM seq_examples ORDER BY RANDOM() LIMIT {int(num_examples * 0.1)}")
  cursor.execute("INSERT INTO train SELECT id, input, target FROM seq_examples WHERE id NOT IN (SELECT id FROM test)")
  cursor.execute("DROP TABLE seq_examples")

  # init database interfaces for training
  seq_tokenizer = T5Tokenizer.from_pretrained(SEQ_MODEL_NAME, legacy=False)
  score_tokenizer = BertTokenizer.from_pretrained(SCORE_MODEL_NAME)
  train_ds = SQLiteDatasetEncoded(connection, "train", ["input", "target"], tokenizer=seq_tokenizer, encode_fn=encode_seq, device=DEVICE)
  test_ds = SQLiteDatasetEncoded(connection, "test", ["input", "target"], tokenizer=seq_tokenizer, encode_fn=encode_seq, device=DEVICE)
  train_dl = DataLoader(dataset=train_ds, shuffle=True, collate_fn=collate_seq, batch_size=BATCH_SIZE, )
  test_dl = DataLoader(dataset=test_ds, shuffle=True, collate_fn=collate_seq, batch_size=BATCH_SIZE)

  # init the models
  seq_model = T5ForConditionalGeneration.from_pretrained(SEQ_MODEL_NAME)
  if os.path.exists(seq_path):
    state_dict = torch.load(seq_path, weights_only=True)
    seq_model.load_state_dict(state_dict)
    print("SEQ2SEQ MODEL LOADED")
  seq_model.to(DEVICE)
  score_model = BertForSequenceClassification.from_pretrained(SCORE_MODEL_NAME, num_labels=4)
  if os.path.exists(score_path):
    state_dict = torch.load(score_path, weights_only=True)
    score_model.load_state_dict(state_dict)
    print("SCORE MODEL LOADED")
  score_model.to(DEVICE)

  # init other training hyper parameters
  dataloaders = {"train": train_dl, "val": test_dl}
  criterion = compute_loss
  optimizer = torch.optim.AdamW(seq_model.parameters(), lr=5e-5)

  # train model
  train_seq2seq(seq_model, score_model, score_tokenizer, score_tokenizer, dataloaders, criterion, optimizer)
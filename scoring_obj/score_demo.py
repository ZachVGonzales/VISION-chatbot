
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
import torch
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_params():
  parser = argparse.ArgumentParser(prog="score_demo.py", 
                                   description="demo the scoring models capabilities")
  parser.add_argument("model_path", help="path to saved model (None if no saved model exists)")
  return parser.parse_args()


def score(model, tokenizer, objective: str):
  encoded_input = tokenizer(objective, return_tensors="pt")
  encoded_input.to(DEVICE)
  with torch.no_grad():
    logits = model(**encoded_input).logits
    logits = torch.sigmoid(logits)
    logits = logits.squeeze(0)
    print(logits)
  return None


if __name__ == "__main__":
  params = init_params()
  model_path = params.model_path

  state_dict = torch.load(model_path, weights_only=True)
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 4)
  model.load_state_dict(state_dict)
  model.to(DEVICE)
  print("model loaded...")

  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  while True:
    objective = input("Enter Objective: ")
    obj_score = score(model, tokenizer, objective)
    break # continue here...

  quit()
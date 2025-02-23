
from transformers import BertTokenizerFast
from utils.models import BERTMultiTaskClassifier
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
    logits = model(**encoded_input)
    abcd_pred = torch.sigmoid(logits["abcd"]).squeeze(0).tolist()
    type_pred = torch.sigmoid(logits["types"]).squeeze(0).tolist()
    bloom_pred = torch.softmax(logits["blooms"]).squeeze(0).tolist()
    print(f"abcd: {abcd_pred}\ntype: {type_pred}\nbloom: {bloom_pred}")
  return None


if __name__ == "__main__":
  params = init_params()
  model_path = params.model_path

  state_dict = torch.load(model_path, weights_only=True)
  classes = {"abcd": (False, ["audience", "behavior", "condition", "degree"]), "types": (False, ["type1", "type2"]), "blooms": (True, ["BT1", "BT2", "BT3", "BT4", "BT5", "BT6"])}
  model = BERTMultiTaskClassifier(classes=classes, device=DEVICE, state_dict=state_dict)

  model.load_state_dict(state_dict)
  model.to(DEVICE)
  print("model loaded...")

  tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

  while True:
    objective = input("Enter Objective: ")
    if objective == "":
      break

    score(model, tokenizer, objective)

  quit()
from transformers import T5Tokenizer, T5ForConditionalGeneration # type: ignore
import torch
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ARCH = "google/flan-t5-base"


def init_params():
  parser = argparse.ArgumentParser(prog="score_demo.py", 
                                   description="demo the scoring models capabilities")
  parser.add_argument("model_path", help="path to saved model (None if no saved model exists)")
  return parser.parse_args()


def generate(model, tokenizer, objective: str):
  encoded_inputs = tokenizer(objective, return_tensors="pt")
  encoded_inputs.to(DEVICE)
  with torch.no_grad():
    seq_ids = model.generate(**encoded_inputs)
    sequences = tokenizer.batch_decode(seq_ids)
    print(sequences)
  return None


if __name__ == "__main__":
  params = init_params()
  model_path = params.model_path

  state_dict = torch.load(model_path, weights_only=True)
  model = T5ForConditionalGeneration.from_pretrained(MODEL_ARCH)
  model.load_state_dict(state_dict)
  model.to(DEVICE)
  print("model loaded...")

  tokenizer = T5Tokenizer.from_pretrained(MODEL_ARCH, legacy=False)

  while True:
    objective = input("Enter Objective: ")
    if objective == "":
      break

    generate(model, tokenizer, objective)

  quit()
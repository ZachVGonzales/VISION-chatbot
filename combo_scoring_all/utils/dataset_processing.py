import torch
from torch.nn.utils.rnn import pad_sequence


def encode_scoring(example, tokenizer, device):
  text = " ".join(example["text"])
  labels = example["labels"]

  with torch.no_grad():
    encodings = tokenizer(
      text,
      max_length = 512,
      padding="max_length",
      truncation=True, 
      return_tensors="pt"
    ).to(device)

  if labels is not None:
    labels_tensor = torch.tensor(labels, dtype=torch.float).to(device)
  else:
    labels_tensor = None

  return {
    "input_ids": encodings.input_ids.squeeze(0).to(device),
    "attention_mask": encodings.attention_mask.squeeze(0).to(device),
    "labels": labels_tensor
  }


def collate_scoring(batch):
  input_ids = [torch.tensor(example["input_ids"]).squeeze() if type(example["input_ids"]) != torch.Tensor else example["input_ids"] for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]).squeeze() if type(example["attention_mask"]) != torch.Tensor else example["attention_mask"] for example in batch]
  labels = [torch.tensor(example["labels"], dtype=torch.float) if type(example["labels"]) != torch.Tensor else example["labels"] for example in batch]

  # Pad sequences to have the same length
  input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) 
  attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  labels = pad_sequence(labels, batch_first=True, padding_value=0)

  return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
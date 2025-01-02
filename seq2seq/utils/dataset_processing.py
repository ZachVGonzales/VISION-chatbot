import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F



def encode_seq(example: dict, tokenizer, device: torch.device) -> dict:
  """
  Encodes an example for the seq2seq model using the given tokenizer.

  Args:
    example: dictionary of tokenizer inputs
    tokenizer: the tokenizer being used to generate model inputs
    device: the device on which the tensor is expected to be returned on
  
  Returns:
    dict: a dictionary of model inputs
  """
  input = " ".join(example["input"])
  target = " ".join(example["target"])

  with torch.no_grad():
    input_encodings = tokenizer(
      input,
      max_length = 512,
      padding="max_length",
      truncation=True, 
      return_tensors="pt"
    ).to(device)
    target_encodings = tokenizer(
      target,
      max_length = 512,
      padding="max_length",
      truncation=True, 
      return_tensors="pt"
    ).to(device)

  return {
    "raw_input": input,
    "raw_target": target,
    "input_ids": input_encodings.input_ids.squeeze(0).to(device),
    "attention_mask": input_encodings.attention_mask.squeeze(0).to(device),
    "target_ids": target_encodings.input_ids.squeeze(0).to(device)
  }


def collate_seq(batch):
  raw_inputs = [example["raw_input"] for example in batch]
  raw_targets = [example["raw_target"] for example in batch]
  input_ids = [torch.tensor(example["input_ids"]).squeeze() if type(example["input_ids"]) != torch.Tensor else example["input_ids"] for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]).squeeze() if type(example["attention_mask"]) != torch.Tensor else example["attention_mask"] for example in batch]
  target_ids = [torch.tensor(example["target_ids"], dtype=torch.float) if type(example["target_ids"]) != torch.Tensor else example["target_ids"] for example in batch]

  # Pad sequences to have the same length
  input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) 
  attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

  return {"input_ids": input_ids, "attention_mask": attention_mask, "target_ids": target_ids, "raw_input": raw_inputs, "raw_target": raw_targets}


def compute_loss(std_loss: torch.Tensor, reward_scores: torch.Tensor):
    """
    Compute a custom loss combining language modeling loss and reward.
    
    Args:
        std: Model output logits.
        rewards: Rewards for the generated sequences.
    
    Returns:
        torch.Tensor: Loss value.
    """
    a_weight = 0.8
    b_weight = 0.2

    ideal_scores = torch.ones_like(reward_scores)
    reward_loss = F.mse_loss(reward_scores, ideal_scores)

    total_loss = a_weight * std_loss + b_weight * reward_loss
    return total_loss

def score_seq(model, tokenizer, sequence, device):
  with torch.no_grad():
    input_encodings = tokenizer(
      sequence,
      truncation=True, 
      return_tensors="pt"
    ).to(device)
    logits = model(**input_encodings).logits
    logits = torch.sigmoid(logits)
    scores = logits.squeeze(0)
  
  return scores
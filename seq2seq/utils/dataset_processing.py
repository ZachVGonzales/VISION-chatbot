import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


def calc_reward(scores: list[float], weights: list[float] = [1.0, 1.0, 1.0, 1.0]) -> float:
  """
  Calculate a reward from classification scores.

  Args:
    scores: list of scores for objective score categories (a, b, c, d)
    weights: optional list of weights for categories
  
  Returns:
    float: A reward value
  """
  return sum(w * s for w, s in zip(weights, scores))


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
    "input_ids": input_encodings.input_ids.squeeze(0).to(device),
    "attention_mask": input_encodings.attention_mask.squeeze(0).to(device),
    "target_ids": target_encodings.input_ids.squeeze(0).to(device)
  }


def collate_seq(batch):
  input_ids = [torch.tensor(example["input_ids"]).squeeze() if type(example["input_ids"]) != torch.Tensor else example["input_ids"] for example in batch]
  attention_mask = [torch.tensor(example["attention_mask"]).squeeze() if type(example["attention_mask"]) != torch.Tensor else example["attention_mask"] for example in batch]
  target_ids = [torch.tensor(example["target_ids"], dtype=torch.float) if type(example["target_ids"]) != torch.Tensor else example["target_ids"] for example in batch]

  # Pad sequences to have the same length
  input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) 
  attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

  return {"input_ids": input_ids, "attention_mask": attention_mask, "target_ids": target_ids}


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, rewards: torch.Tensor):
    """
    Compute a custom loss combining language modeling loss and reward.
    
    Args:
        logits: Model output logits.
        labels: Ground truth token IDs.
        rewards: Rewards for the generated sequences.
    
    Returns:
        torch.Tensor: Loss value.
    """
    # Cross-entropy loss for language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

    # Reshape to batch size
    batch_size = labels.size(0)
    lm_loss = lm_loss.view(batch_size, -1).mean(dim=1)
    
    # Reward-weighted loss
    reward_loss = lm_loss * rewards
    return reward_loss.mean()


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
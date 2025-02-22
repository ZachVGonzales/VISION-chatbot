import torch
import torch.nn as nn
from transformers import BertModel


class BERTMultiTaskClassifier(nn.Module):
  """A BERT sequence classification model that can be used for multi-task 
  classification (both mutually exclusive and inclusive 
  labeling).

  :param classes: The classes that the model shall support, should be of the format 
  {"class1": (mutually exclusive T/F, ["label1", "label2", ...]), "class2": ...}
  :param device: The device on which the computation of the model shall take place"""
  def __init__(self, classes: dict, device: torch.device, state_dict=None):
    super().__init__()
    self.device = device
    self.bert_layer = BertModel.from_pretrained("bert-base-uncased").to(device)
    if state_dict:
      self.bert_layer.load_state_dict(state_dict)
    self.hidden_size = self.bert_layer.config.hidden_size
    self.classes = classes
    self.classifiers = {}
    
    for k, v in classes.items():
      labels = v[1]
      classifier = nn.Linear(self.hidden_size, len(labels)).to(self.device)
      self.classifiers.setdefault(k, classifier)
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert_layer.forward(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output # this comes from the CLS token
    
    logits = {k: self.classifiers[k](pooled_output) for k in self.classes.keys()}
    return logits
  
  def compute_loss(self, logits: dict, labels: dict):
    # True if first 4 labels are all zero
    #print(logits)
    #print(labels)
    mask_condition_types = ((labels["abcd"].sum(dim=1) == 0).bool() | (labels["blooms"].sum(dim=1) == 0).bool())
    mask_condition_blooms = (labels["blooms"].sum(dim=1) == 0)
    mask_types = torch.ones_like(labels["types"])
    mask_blooms = torch.ones_like(labels["blooms"])
    mask_abcd = torch.ones_like(labels["abcd"])

    # Zero out last 2 categories if condition met
    mask_types[mask_condition_types, :] = 0  
    mask_blooms[mask_condition_blooms, :] = 0

    exclusive_loss_fn = nn.CrossEntropyLoss()
    independent_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = torch.tensor(0.0, device=self.device)

    for k, v in self.classes.items():
      if v[0]:
        loss = exclusive_loss_fn(logits[k], torch.argmax(labels[k], dim=1))
      else:
        loss = independent_loss_fn(logits[k], labels[k])
      
      if k == "blooms":
        loss = (mask_blooms * loss).sum()
      elif k == "type":
        loss = (mask_types * loss).sum()
      
      total_loss += loss 
    
    total_loss = total_loss / (mask_types.sum() + mask_blooms.sum() + mask_abcd.sum())
    return total_loss
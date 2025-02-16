import csv
import random

def get_contents(path: str):
  contents = []
  with open(path, "r") as file:
    for line in file:
      line = line.strip()
      contents.append(line)
  return contents


if __name__ == "__main__":
  file = open("objectives.csv", "r")
  rows = csv.reader(file)

  types_idxs = {}
  for i, row in enumerate(rows):
    score = [int(row[1]), int(row[2]), int(row[3]), int(row[4])].__str__()
    types_idxs.setdefault(score, []).append(i+1)

  min_examples = min(types_idxs, key=lambda k: len(types_idxs[k]))
  print(f"min examples found is {min_examples}: {len(types_idxs[min_examples])}")

  print("-------DONE-------")
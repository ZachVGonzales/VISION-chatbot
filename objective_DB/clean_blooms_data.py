import pandas as pd
import csv


common_mapping1 = {"BT1": [1,0,0,0,0,0], "BT2": [0,1,0,0,0,0], "BT3": [0,0,1,0,0,0], "BT4": [0,0,0,1,0,0], "BT5": [0,0,0,0,1,0], "BT6": [0,0,0,0,0,1]}
common_mapping2 = {1: [1,0,0,0,0,0], 2: [0,1,0,0,0,0], 3: [0,0,1,0,0,0], 4: [0,0,0,1,0,0], 5: [0,0,0,0,1,0], 6: [0,0,0,0,0,1]}


data_file1 = open("blooms_taxonomy_dataset.csv", "r")
data_file2 = open("Exam_Questions_According_to_BT_Levels.csv", "r")
outfile = open("blooms_combined.csv", "w")
writer = csv.writer(outfile)
writer.writerow(["text", "BT1", "BT2", "BT3", "BT4", "BT5", "BT6"])

df1 = pd.read_csv(data_file1)
df2 = pd.read_csv(data_file2)

objectives = [objective if ("you" not in objective) and ("You" not in objective) else None for objective in df1["Questions"]]
labels = [common_mapping1[label] for label in df1["Category"]]

for objective, label in zip(objectives, labels):
  if objective is None:
    continue
  row = [objective] + label
  writer.writerow(row)

objectives = [objective if (not isinstance(objective, float)) and ("you" not in objective) and ("You" not in objective) else None for objective in df2["Questions"]]
labels = [label for label in df2["Category"]]

for objective, label in zip(objectives, labels):
  if objective is None:
    continue
  label = int(label)
  label = common_mapping2[label]
  row = [objective] + label
  writer.writerow(row)
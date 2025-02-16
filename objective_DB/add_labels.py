import pandas as pd
import csv


if __name__ == "__main__":
  outfile = open("b_labled.csv", "w")
  writer = csv.writer(outfile)
  writer.writerow(["behvior", "type"])

  df = pd.read_csv("objective_types.csv")
  b_txt = open("b.txt", "r")
  for behavior in b_txt:
    behavior = behavior.strip()
    for i, objective in enumerate(df["objective"]):
      #print(behavior, objective)
      if behavior in objective:
        writer.writerow([behavior, df["type"][i]])
        break
    if behavior in objective:
      continue
    print(f"could not find label for {behavior}")
import csv
import random
import math


HEADER = ["input_objective", "audience_score", "behavior_score", "condition_score", "degree_score", "new_objective"]
AB_CON = ["shall", "will", "should", "must", "needs to", "is obliged to"]
CONDITION_PRES = ["given", "without", "with", "Consulting", "lacking"]
SUBJECTS = ["cat", "programmer", "alien", "teacher", "robot", "house"]
ADJS = ["red", "noisy", "smelly", "helpful"]


def write_objectives(audience, behavior, condition, degree):
  abcon = random.choice(AB_CON)
  cond_pre = random.choice(CONDITION_PRES)

  # Generate all pos examples from data
  ab = " ".join([audience, abcon, behavior])
  cb = " ".join([cond_pre, condition, behavior])
  bc = " ".join([behavior, cond_pre, condition])
  bd = " ".join([behavior, degree])
  db = " ".join([degree, behavior])
  abc = " ".join([audience, abcon, behavior, cond_pre, condition])
  acb = " ".join([audience + ",", cond_pre, condition + ",", abcon, behavior])
  cab = " ".join([cond_pre, condition, audience, abcon, behavior])
  abd = " ".join([audience, abcon, behavior, degree])
  adb = " ".join([audience + ",", abcon, degree, behavior])
  dab = " ".join([degree + ",", audience, abcon, behavior])
  cbd = " ".join([cond_pre, condition, behavior, degree])
  dbc = " ".join([degree, behavior, cond_pre, condition])
  bcd = " ".join([behavior, cond_pre, condition, degree])
  bdc = " ".join([behavior, degree, cond_pre, condition])
  abcd = " ".join([audience, abcon, behavior, cond_pre, condition, degree])
  cabd = " ".join([cond_pre, condition, audience, abcon, behavior, degree])
  cdab = " ".join([cond_pre, condition, degree + ",", audience, abcon, behavior])
  dcab = " ".join([degree, cond_pre, condition, audience, abcon, behavior])
  dabc = " ".join([degree + ",", audience, abcon, behavior, cond_pre, condition])
  cadb = " ".join([cond_pre, condition + ",", audience + ",", degree, abcon, behavior])
  dacb = " ".join([degree + ",", audience + ",", cond_pre, condition, abcon, behavior])

  # negative examples
  ac = " ".join([audience, cond_pre, condition])
  ca = " ".join([cond_pre, condition, audience])
  ad = " ".join([audience, degree])
  da = " ".join([degree, audience])
  acd = " ".join([audience, cond_pre, condition, degree])
  adc = " ".join([audience, degree, cond_pre, condition])
  dac = " ".join([degree, audience, cond_pre, condition])
  dca = " ".join([degree, cond_pre, condition, audience])
  cda = " ".join([cond_pre, condition, degree, audience])
  cad = " ".join([cond_pre, condition, audience, degree])

  # Write data to a CSV file
  try:
    with open("objectives.csv", "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([audience, 0, 0, 0, 0, cabd])
      writer.writerow([cond_pre + " " + condition, 0, 0, 0, 0, cabd])
      writer.writerow([degree, 0, 0, 0, 0, cabd])
      writer.writerow([ac, 0, 0, 0, 0, cabd])
      writer.writerow([ca, 0, 0, 0, 0, cabd])
      writer.writerow([ad, 0, 0, 0, 0, cabd])
      writer.writerow([da, 0, 0, 0, 0, cabd])
      writer.writerow([acd, 0, 0, 0, 0, cabd])
      writer.writerow([adc, 0, 0, 0, 0, cabd])
      writer.writerow([dac, 0, 0, 0, 0, cabd])
      writer.writerow([dca, 0, 0, 0, 0, cabd])
      writer.writerow([cda, 0, 0, 0, 0, cabd])
      writer.writerow([cad, 0, 0, 0, 0, cabd])

      writer.writerow([behavior, 0, 1, 0, 0, cabd])
      writer.writerow([ab, 1, 1, 0, 0, cabd])
      writer.writerow([cb, 0, 1, 1, 0, cabd])
      writer.writerow([bc, 0, 1, 1, 0, cabd])
      writer.writerow([bd, 0, 1, 0, 1, cabd])
      writer.writerow([db, 0, 1, 0, 1, cabd])
      writer.writerow([abc, 1, 1, 1, 0, cabd])
      writer.writerow([acb, 1, 1, 1, 0, cabd])
      writer.writerow([cab, 1, 1, 1, 0, cabd])
      writer.writerow([abd, 1, 1, 0, 1, cabd])
      writer.writerow([adb, 1, 1, 0, 1, cabd])
      writer.writerow([dab, 1, 1, 0, 1, cabd])
      writer.writerow([cbd, 0, 1, 1, 1, cabd])
      writer.writerow([bcd, 0, 1, 1, 1, cabd])
      writer.writerow([dbc, 0, 1, 1, 1, cabd])
      writer.writerow([bdc, 0, 1, 1, 1, cabd])
      writer.writerow([abcd, 1, 1, 1, 1, cabd])
      writer.writerow([cabd, 1, 1, 1, 1, cabd])
      writer.writerow([cdab, 1, 1, 1, 1, cabd])
      writer.writerow([dcab, 1, 1, 1, 1, cabd])
      writer.writerow([dabc, 1, 1, 1, 1, cabd])
      writer.writerow([cadb, 1, 1, 1, 1, cabd])
      writer.writerow([dacb, 1, 1, 1, 1, cabd])
  except Exception as e:
    print("Error", f"Failed to save data: {e}")


def get_contents(path: str):
  contents = []
  with open(path, "r") as file:
    for line in file:
      line = line.strip()
      contents.append(line)
  return contents


if __name__ == "__main__":
  # init csv file
  with open("objectives.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(HEADER)
    file.close()

  # get the content of the feader files
  audiences = get_contents("a.txt")
  behaviors = get_contents("b.txt")
  conditions = get_contents("c.txt")
  degrees = get_contents("d.txt")

  # get the random selection vals
  arand = 3
  crand = 3
  drand = 2

  # generate at least one example / behavior
  for i, behavior in enumerate(behaviors):
    print(f"processing behavior {i}/{len(behaviors)}")

    # select random examples from other categories
    selectedA = random.sample(audiences, arand)
    selectedC = random.sample(conditions, crand)
    selectedD = random.sample(degrees, drand)

    # go through all possible generating examples for each
    for audience in selectedA:
      for condition in selectedC:
        for degree in selectedD:
          write_objectives(audience, behavior, condition, degree)

  print("-------DONE-------")
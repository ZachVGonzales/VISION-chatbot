import csv
import random
import math


HEADER = ["input_objective", "audience_score", "behavior_score", "condition_score", "degree_score", "new_objective"]
AB_CON = ["shall", "will", "should", "must", "needs to", "is obliged to"]
CONDITION_PRES = ["given", "without", "with", "Consulting", "lacking"]


def write_objectives(audience, behavior, condition, degree):
  abcon = random.choice(AB_CON)
  cond_pre = random.choice(CONDITION_PRES)
  possible_exs = []

  # Generate all pos examples from data and store in set
  possible_exs.append((" ".join([audience, abcon, behavior]), [1,1,0,0]))
  possible_exs.append((" ".join([cond_pre, condition, behavior]), [0,1,1,0]))
  possible_exs.append((" ".join([behavior, cond_pre, condition]), [0,1,1,0]))
  possible_exs.append((" ".join([behavior, degree]), [0,1,0,1]))
  possible_exs.append((" ".join([degree, behavior]), [0,1,0,1]))
  possible_exs.append((" ".join([audience, abcon, behavior, cond_pre, condition]), [1,1,1,0]))
  possible_exs.append((" ".join([audience + ",", cond_pre, condition + ",", abcon, behavior]), [1,1,1,0]))
  possible_exs.append((" ".join([cond_pre, condition, audience, abcon, behavior]), [1,1,1,0]))
  possible_exs.append((" ".join([audience, abcon, behavior, degree]), [1,1,0,1]))
  possible_exs.append((" ".join([audience + ",", abcon, degree, behavior]), [1,1,0,1]))
  possible_exs.append((" ".join([degree + ",", audience, abcon, behavior]), [1,1,0,1]))
  possible_exs.append((" ".join([cond_pre, condition, behavior, degree]), [0,1,1,1]))
  possible_exs.append((" ".join([degree, behavior, cond_pre, condition]), [0,1,1,1]))
  possible_exs.append((" ".join([behavior, cond_pre, condition, degree]), [0,1,1,1]))
  possible_exs.append((" ".join([behavior, degree, cond_pre, condition]), [0,1,1,1]))
  possible_exs.append((" ".join([audience, abcon, behavior, cond_pre, condition, degree]), [1,1,1,1]))
  possible_exs.append((" ".join([cond_pre, condition, audience, abcon, behavior, degree]), [1,1,1,1]))
  possible_exs.append((" ".join([cond_pre, condition, degree + ",", audience, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree, cond_pre, condition, audience, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree + ",", audience, abcon, behavior, cond_pre, condition]), [1,1,1,1]))
  possible_exs.append((" ".join([cond_pre, condition + ",", audience + ",", degree, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree + ",", audience + ",", cond_pre, condition, abcon, behavior]), [1,1,1,1]))

  cabd = " ".join([cond_pre, condition, audience, abcon, behavior, degree])

  # Write data to a CSV file
  try:
    with open("objectives.csv", "a", newline="") as file:
      writer = csv.writer(file)
      selection = random.sample(possible_exs, 4)
      for example in selection:
        writer.writerow((example[0], example[1][0], example[1][1], example[1][2], example[1][3], cabd))
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
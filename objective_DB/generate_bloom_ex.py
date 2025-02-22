import csv
import random
import math
import string
import pandas as pd


HEADER = ["objective", "BT1", "BT2", "BT3", "BT4", "BT5", "BT6"]
AB_CON = ["shall", "will", "should", "must", "needs to", "is obliged to"]
CONDITION_PRES = ["given", "without", "with", "Consulting", "lacking"]


random_words = [
  "the", "system", "model", "banana", "toaster", "lorem", "ipsum", "why", "because", "error", "debug",
  "running", "circles", "thinking", "unknown", "random", "words", "meaningless", "something", "nothing",
  "purple", "elephant", "spaceship", "algorithm", "paradox", "cheese", "matrix", "illusion", "computer",
  "trainee", "student", "maintain", "with", "inspect", "flow", "typical"
]

# Function to generate random gibberish words
def random_gibberish_word(length=5):
  return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

# Function to generate a completely random gibberish sentence
def generate_gibberish_sentence():
  sentence_length = random.randint(5, 15)
  return ' '.join(random_gibberish_word(random.randint(3, 8)) for _ in range(sentence_length))

# Function to generate repetitive nonsense
def generate_repetitive_text():
  word = random.choice(random_words)
  return ' '.join([word] * random.randint(5, 15))

# Function to generate broken and unfinished sentences
def generate_broken_sentence():
  sentence = ' '.join(random.choice(random_words) for _ in range(random.randint(3, 10)))
  return sentence[:random.randint(5, len(sentence))]  # Randomly cut it off

# Function to generate structured nonsense
def generate_structured_nonsense():
  templates = [
    "The {word1} of the {word2} is the {word3} of the {word4}",
    "If the {word1} is {word2}, then the {word3} must be {word4}",
    "Without the {word1}, the {word2} cannot {word3}",
    "{word1}, {word2}, {word3}, and {word4} walk into a {word5}",
    "Repeating {word1} over and over and over and over again"
  ]
  template = random.choice(templates)
  return template.format(
    word1=random.choice(random_words),
    word2=random.choice(random_words),
    word3=random.choice(random_words),
    word4=random.choice(random_words),
    word5=random.choice(random_words),
  )

# Function to generate a mix of gibberish, repetitive text, and broken sentences
def generate_null_example():
  generators = [
    generate_gibberish_sentence,
    generate_repetitive_text,
    generate_broken_sentence,
    generate_structured_nonsense
  ]
  return random.choice(generators)()

def write_objectives(audience, behavior, condition, degree, label: list):
  abcon = random.choice(AB_CON)
  #cond_pre = random.choice(CONDITION_PRES)
  possible_exs = []

  # Generate all pos examples from data and store in set
  possible_exs.append((" ".join([behavior]), [0,1,0,0]))
  possible_exs.append((" ".join([audience, abcon, behavior]), [1,1,0,0]))
  possible_exs.append((" ".join([condition, behavior]), [0,1,1,0]))     
  possible_exs.append((" ".join([behavior, condition]), [0,1,1,0]))
  possible_exs.append((" ".join([behavior, degree]), [0,1,0,1]))
  possible_exs.append((" ".join([degree, behavior]), [0,1,0,1]))
  possible_exs.append((" ".join([audience, abcon, behavior, condition]), [1,1,1,0]))
  possible_exs.append((" ".join([audience + ",", condition + ",", abcon, behavior]), [1,1,1,0]))
  possible_exs.append((" ".join([condition + ",", audience, abcon, behavior]), [1,1,1,0]))
  possible_exs.append((" ".join([audience, abcon, behavior, degree]), [1,1,0,1]))
  possible_exs.append((" ".join([audience + ",", abcon, degree, behavior]), [1,1,0,1]))
  possible_exs.append((" ".join([degree + ",", audience, abcon, behavior]), [1,1,0,1]))
  possible_exs.append((" ".join([condition, behavior, degree]), [0,1,1,1]))
  possible_exs.append((" ".join([degree + ",", behavior, condition]), [0,1,1,1]))
  possible_exs.append((" ".join([behavior, condition +",", degree]), [0,1,1,1]))
  possible_exs.append((" ".join([behavior, degree + ",", condition]), [0,1,1,1]))
  possible_exs.append((" ".join([audience, abcon, behavior, condition + ",", degree]), [1,1,1,1]))
  possible_exs.append((" ".join([condition, audience, abcon, behavior, degree]), [1,1,1,1]))
  possible_exs.append((" ".join([condition, degree + ",", audience, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree + ",", condition + ",", audience, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree + ",", audience, abcon, behavior, condition]), [1,1,1,1]))
  possible_exs.append((" ".join([condition + ",", audience + ",", degree, abcon, behavior]), [1,1,1,1]))
  possible_exs.append((" ".join([degree + ",", audience + ",", condition, abcon, behavior]), [1,1,1,1]))

  cabd = " ".join([condition, audience, abcon, behavior, degree])

  # Write data to a CSV file
  try:
    with open("blooms_objectives.csv", "a", newline="") as file:
      writer = csv.writer(file)
      selection = random.sample(possible_exs, 4)
      for example in selection:
        writer.writerow([example[0]]+label)
  except Exception as e:
    print("Error", f"Failed to save data: {e}")


def write_null(null_ex):
  try:
    with open("blooms_objectives.csv", "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow((null_ex, 0, 0, 0, 0, 0, 0))
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
  with open("blooms_objectives.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(HEADER)
    file.close()

  # get the content of the feader files
  audiences = get_contents("a.txt")
  conditions = get_contents("c.txt")
  degrees = get_contents("d.txt")
  null_exs = get_contents("null_examples.txt")

  # get the random selection vals
  num_rand = 2

  # get the behaviors and labels in a dataframe
  bl_df = pd.read_csv("blooms_combined.csv")
  num_b = len(bl_df["text"])

  # generate at least one example / behavior
  for i, behavior in enumerate(random.sample(bl_df["text"].to_list(), num_b)):
    print(f"processing behavior {i}/{num_b}")

    # go through all possible generating examples for each
    for _ in range(num_rand):
      audience = random.sample(audiences, 1)[0]
      condition = random.sample(conditions, 1)[0]
      degree = random.sample(degrees, 1)[0]
      write_objectives(audience, behavior, condition, degree, [bl_df["BT1"][i], bl_df["BT2"][i], bl_df["BT3"][i], bl_df["BT4"][i], bl_df["BT5"][i], bl_df["BT6"][i]])

  for null_ex in null_exs:
    write_null(null_ex)

  for _ in range(math.ceil((num_rand)*(len(bl_df["text"])/10))):
    write_null(generate_null_example())

  print("-------DONE-------")
import tkinter as tk
import csv
from tkinter import messagebox
import random


HEADER = ["input_objective", "audience_score", "behavior_score", "condition_score", "degree_score", "new_objective"]
AB_CON = ["shall", "will", "should", "must", "needs to", "is obliged to"]
CONDITION_PRES = ["given", "without", "with", "Consulting", "lacking"]
SUBJECTS = ["cat", "programmer", "alien", "teacher", "robot", "house"]
ADJS = ["red", "noisy", "smelly", "helpful"]


def write_objectives(text_boxes: list[tk.Entry]):
  audience = text_boxes[0].get()
  behavior = text_boxes[1].get()
  condition = text_boxes[2].get()
  degree = text_boxes[3].get()

  # Generate all pos examples from data
  ab = " ".join([audience, random.choice(AB_CON), behavior])
  cb = " ".join([random.choice(CONDITION_PRES), condition, behavior])
  bc = " ".join([behavior, random.choice(CONDITION_PRES), condition])
  bd = " ".join([behavior, degree])
  db = " ".join([degree, behavior])
  abc = " ".join([audience, random.choice(AB_CON), behavior, random.choice(CONDITION_PRES), condition])
  acb = " ".join([audience + ",", random.choice(CONDITION_PRES), condition + ",", random.choice(AB_CON), behavior])
  cab = " ".join([random.choice(CONDITION_PRES), condition, audience, random.choice(AB_CON), behavior])
  abd = " ".join([audience, random.choice(AB_CON), behavior, degree])
  adb = " ".join([audience + ",", random.choice(AB_CON), degree, behavior])
  dab = " ".join([degree + ",", audience, random.choice(AB_CON), behavior])
  cbd = " ".join([random.choice(CONDITION_PRES), condition, behavior, degree])
  dbc = " ".join([degree, behavior, random.choice(CONDITION_PRES), condition])
  bcd = " ".join([behavior, random.choice(CONDITION_PRES), condition, degree])
  bdc = " ".join([behavior, degree, random.choice(CONDITION_PRES), condition])
  abcd = " ".join([audience, random.choice(AB_CON), behavior, random.choice(CONDITION_PRES), condition, degree])
  cabd = " ".join([random.choice(CONDITION_PRES), condition, audience, random.choice(AB_CON), behavior, degree])
  cdab = " ".join([random.choice(CONDITION_PRES), condition, degree + ",", audience, random.choice(AB_CON), behavior])
  dcab = " ".join([degree, random.choice(CONDITION_PRES), condition, audience, random.choice(AB_CON), behavior])
  dabc = " ".join([degree + ",", audience, random.choice(AB_CON), behavior, random.choice(CONDITION_PRES), condition])
  cadb = " ".join([random.choice(CONDITION_PRES), condition + ",", audience + ",", degree, random.choice(AB_CON), behavior])
  dacb = " ".join([degree + ",", audience + ",", random.choice(CONDITION_PRES), condition, random.choice(AB_CON), behavior])

  # negative examples
  ac = " ".join([audience, random.choice(CONDITION_PRES), condition])
  ca = " ".join([random.choice(CONDITION_PRES), condition, audience])
  ad = " ".join([audience, degree])
  da = " ".join([degree, audience])
  acd = " ".join([audience, random.choice(CONDITION_PRES), condition, degree])
  adc = " ".join([audience, degree, random.choice(CONDITION_PRES), condition])
  dac = " ".join([degree, audience, random.choice(CONDITION_PRES), condition])
  dca = " ".join([degree, random.choice(CONDITION_PRES), condition, audience])
  cda = " ".join([random.choice(CONDITION_PRES), condition, degree, audience])
  cad = " ".join([random.choice(CONDITION_PRES), condition, audience, degree])

  # make some giberish
  gib1 = " ".join([random.choice(AB_CON), random.choice(ADJS), random.choice(SUBJECTS)])
  gib2 = " ".join([random.choice(ADJS), random.choice(SUBJECTS)])

  # Write data to a CSV file
  try:
    with open("objectives_seq.csv", "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([gib1, 0, 0, 0, 0, cabd])
      writer.writerow([gib2, 0, 0, 0, 0, cabd])
      writer.writerow([audience, 0, 0, 0, 0, cabd])
      writer.writerow([random.choice(CONDITION_PRES) + " " + condition, 0, 0, 0, 0, cabd])
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
    messagebox.showinfo("Success", "Data saved to output.csv")
  except Exception as e:
    messagebox.showerror("Error", f"Failed to save data: {e}")


if __name__ == "__main__":
  # init csv file
  with open("objectives.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(HEADER)
    file.close()

  root = tk.Tk()
  root.title("CSV Writer")
  root.geometry("400x300")

  # Create text boxes
  text_box1 = tk.Entry(root, width=50)
  text_box2 = tk.Entry(root, width=50)
  text_box3 = tk.Entry(root, width=50)
  text_box4 = tk.Entry(root, width=50)

  # Create labels
  label1 = tk.Label(root, text="Audience: (the trainee)")
  label2 = tk.Label(root, text="Behavior: (do this thing)")
  label3 = tk.Label(root, text="Condition: (conditional object)")
  label4 = tk.Label(root, text="Degree: (Within this constraint) OR (with accuracy _)")

  # Add text boxes to the window
  label1.pack(pady=2)
  text_box1.pack(pady=10)
  label2.pack(pady=2)
  text_box2.pack(pady=10)
  label3.pack(pady=2)
  text_box3.pack(pady=10)
  label4.pack(pady=2)
  text_box4.pack(pady=10)
  boxes = [text_box1, text_box2, text_box3, text_box4]

  # Create and add the button
  save_button = tk.Button(root, text="Save to CSV", command=lambda: write_objectives(boxes))
  save_button.pack(pady=20)

  # Run the Tkinter event loop
  root.mainloop()
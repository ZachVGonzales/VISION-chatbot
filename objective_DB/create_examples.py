import tkinter as tk
import csv
from tkinter import messagebox
import random


HEADER = ["input_objective", "audience_score", "behavior_score", "condition_score", "degree_score", "new_objective"]
AB_CON = ["shall", "will", "should", "must", "needs to", "is obliged to"]
CONDITION_PRES = ["given this", "without", "with", "Consulting this", "lacking"]
SUBJECTS = ["cat", "programmer", "alien", "teacher", "robot", "house"]
ADJS = ["red", "noisy", "smelly", "helpful"]
PREF = ["this is", "there is", "blah"]


def write_objectives(text_boxes: list[tk.Entry]):
  audience = text_boxes[0].get()
  behavior = text_boxes[1].get()
  condition = text_boxes[2].get()
  degree = text_boxes[3].get()

  # Generate all pos examples from data
  ab = " ".join([audience, random.choice(AB_CON), behavior])
  ac = " ".join([random.choice(CONDITION_PRES), condition, audience])
  ad = " ".join([audience, degree])
  cb = " ".join([random.choice(CONDITION_PRES), condition, behavior])
  bc = " ".join([behavior, random.choice(CONDITION_PRES), condition])
  bd = " ".join([behavior, degree])
  db = " ".join([degree, behavior])
  abc = " ".join([audience, random.choice(AB_CON), behavior, random.choice(CONDITION_PRES), condition])
  cab = " ".join([random.choice(CONDITION_PRES), condition, audience, random.choice(AB_CON), behavior])
  abd = " ".join([audience, random.choice(AB_CON), behavior, degree])
  adb = " ".join([audience, random.choice(AB_CON), degree, behavior])
  acd = " ".join([random.choice(CONDITION_PRES), condition, audience, degree])
  cbd = " ".join([random.choice(CONDITION_PRES), condition, behavior, degree])
  bcd = " ".join([behavior, random.choice(CONDITION_PRES), condition, degree])
  abcd = " ".join([random.choice(CONDITION_PRES), condition, audience, random.choice(AB_CON), behavior, degree])

  # make some giberish
  gib1 = " ".join([random.choice(AB_CON), random.choice(ADJS), random.choice(SUBJECTS)])
  gib2 = " ".join([random.choice(ADJS), random.choice(SUBJECTS)])

  # Write data to a CSV file
  try:
    with open("objectives.csv", "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([gib1, 0, 0, 0, 0, abcd])
      writer.writerow([gib2, 0, 0, 0, 0, abcd])
      writer.writerow([audience, 1, 0, 0, 0, abcd])
      writer.writerow([behavior, 0, 1, 0, 0, abcd])
      writer.writerow([random.choice(CONDITION_PRES) + " " + condition, 0, 0, 1, 0, abcd])
      writer.writerow([degree, 0, 0, 0, 1, abcd])
      writer.writerow([ab, 1, 1, 0, 0, abcd])
      writer.writerow([ac, 1, 0, 1, 0, abcd])
      writer.writerow([ad, 1, 0, 0, 1, abcd])
      writer.writerow([cb, 0, 1, 1, 0, abcd])
      writer.writerow([bc, 0, 1, 1, 0, abcd])
      writer.writerow([bd, 0, 1, 0, 1, abcd])
      writer.writerow([db, 0, 1, 0, 1, abcd])
      writer.writerow([abc, 1, 1, 1, 0, abcd])
      writer.writerow([cab, 1, 1, 1, 0, abcd])
      writer.writerow([abd, 1, 1, 0, 1, abcd])
      writer.writerow([adb, 1, 1, 0, 1, abcd])
      writer.writerow([acd, 1, 0, 1, 1, abcd])
      writer.writerow([cbd, 0, 1, 1, 1, abcd])
      writer.writerow([bcd, 0, 1, 1, 1, abcd])
      writer.writerow([abcd, 1, 1, 1, 1, abcd])
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
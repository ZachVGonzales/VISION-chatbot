import sqlite3
import re


if __name__ == "__main__":
  pattern = r'\x1e(.*?)\x1e'
  out_file = open("objectives.txt", 'w')
  objectives = []

  with open("objectives.OB", 'r') as obj_file:
    for line in obj_file:
      match = re.search(pattern, line)
      if match:
        objective = match.group(1)
        if objective not in objectives:
          out_file.write(f"{objective}\n")
          objectives.append(objective)
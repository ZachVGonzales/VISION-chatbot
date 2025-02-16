from striprtf.striprtf import rtf_to_text
from charset_normalizer import detect
import csv
import re


def process_text(text: str) -> str:
  """Process a chunk of text, convert the RTF to plain text if needed"""
  try:
    return rtf_to_text(text)
  except Exception as e:
    print(f"Error converting RTF text: {e}")
    return text


def clean_text(text: str) -> str:
  """Clean a chunk of plain text, strip ends and remove any internal newlines"""
  return re.sub(r'\s+', ' ', text).strip()


if __name__ == "__main__":
  outfile = open("objective_types.csv", 'w')
  writer = csv.writer(outfile)
  writer.writerow(["objective", "type"])

  with open("CT MECHANICAL MAINTENANCE PROG_data.ob", 'rb') as raw_file:
    raw_data = raw_file.read()
    encoding_info = detect(raw_data)
    file_encoding = encoding_info['encoding']

  print(f"detected encoding: {file_encoding}")

  with open("CT MECHANICAL MAINTENANCE PROG_data.ob", 'r', encoding=file_encoding) as file:
    buffer = ""
    
    for line in file:
      buffer += line

      if '\x1F' in buffer:
        entries = buffer.split('\x1F')
        buffer = entries.pop()

        for entry in entries:
          fields = entry.split('\x1E')
          
          if len(fields) >= 8:
            objective = fields[3]
            obj_type = fields[7]
            objective = clean_text(process_text(objective))
            if obj_type != "Organizer":
              writer.writerow([objective, obj_type])
    
    if buffer.strip():
      fields = buffer.split('\x1E')
      if len(fields) >= 8:
        objective = fields[3]
        objective = clean_text(process_text(objective))
        if obj_type != "Organizer":
          writer.writerow([objective, obj_type])
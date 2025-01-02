import argparse
from striprtf.striprtf import rtf_to_text # type: ignore
from charset_normalizer import detect


def init_params():
  parser = argparse.ArgumentParser(prog="init_obj_db.py", 
                                   description="init the training database for objectives")
  parser.add_argument("ob_path", help="path to the training database")
  parser.add_argument("out_path", help="path to output file")
  return parser.parse_args()


def process_text(text: str):
  """Process a chunk of text, convert the RTF to plain text if needed"""
  try:
    return rtf_to_text(text)
  except Exception as e:
    print(f"Error converting RTF text: {e}")
    return text


if __name__ == "__main__":
  params = init_params()
  ob_path = params.ob_path
  out_path = params.out_path

  outfile = open(out_path, 'w')

  with open(ob_path, 'rb') as raw_file:
    raw_data = raw_file.read()
    encoding_info = detect(raw_data)
    file_encoding = encoding_info['encoding']

  print(f"detected encoding: {file_encoding}")

  with open(ob_path, 'r', encoding=file_encoding) as file:
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
            objective = process_text(objective)
            if obj_type != "Organizer":
              outfile.write(f"{objective}")
    
    if buffer.strip():
      fields = buffer.split('\x1E')
      if len(fields) >= 8:
        objective = fields[3]
        objective = process_text(objective)
        if obj_type != "Organizer":
          outfile.write(f"{objective}")
  
  outfile.close()
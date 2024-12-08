import nltk # type: ignore
from nltk.corpus import wordnet # type: ignore
import random
import sqlite3
import json
from transformers import MarianMTModel, MarianTokenizer # type: ignore



NUM_DUP = 2


# need nltk to use this
def rand_synonym_replacement(words: list[str], p_synonym: float = 0.50) -> list[str]:
  try: # check if wordnet is available, if not then do not attempt replacement
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
  except:
    return words

  new_words = []
  for word in words:
    synonyms = set()
    for syn in wordnet.synsets(word, lang='eng'):
      for lemma in syn.lemmas():
        synonyms.add(lemma.name().replace('_', ' '))
    if synonyms and random.random() < p_synonym:
      new_words.append(random.choice(list(synonyms)))
    else:
      new_words.append(word)
  return new_words


def translate(words: list[str], model, tokenizer) -> list[str]:
  text = " ".join(words)
  inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
  translated = model.generate(inputs, max_length=512)
  return tokenizer.decode(translated[0], skip_special_tokens=True).split()


if __name__ == "__main__":

  # Initialize the model and tokenizer for translation
  model_name_forward = "Helsinki-NLP/opus-mt-en-fr"
  tokenizer_forward = MarianTokenizer.from_pretrained(model_name_forward)
  model_forward = MarianMTModel.from_pretrained(model_name_forward)

  # Back to English
  model_name_back = "Helsinki-NLP/opus-mt-fr-en"
  tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
  model_back = MarianMTModel.from_pretrained(model_name_back)

  # database connection
  conn = sqlite3.connect('objectives.db')
  cursor = conn.cursor()

  # create expanded table if does not already exist
  cursor.execute("CREATE TABLE IF NOT EXISTS augmented_objectives (id INTEGER PRIMARY KEY AUTOINCREMENT, input_objective TEXT NOT NULL, audience_score INTEGER NOT NULL, behavior_score INTEGER NOT NULL, condition_score INTEGER NOT NULL, degree_score INTEGER NOT NULL, new_objective TEXT NOT NULL)")

  # get original objectives 
  cursor.execute("SELECT * FROM objectives")
  rows = cursor.fetchall()

  for row in rows:
    in_obj = json.loads(row[1])
    new_obj = json.loads(row[6])

    for _ in range(NUM_DUP):
      in_obj = translate(in_obj, model_forward, tokenizer_forward)
      in_obj = translate(in_obj, model_back, tokenizer_back)
      new_obj = translate(new_obj, model_forward, tokenizer_forward)
      new_obj = translate(new_obj, model_back, tokenizer_back)
      cursor.execute("INSERT INTO augmented_objectives (input_objective, audience_score, behavior_score, condition_score, degree_score, new_objective) VALUES (?, ?, ?, ?, ?, ?)", (json.dumps(in_obj), row[2], row[3], row[4], row[5], json.dumps(new_obj)))

      in_obj = rand_synonym_replacement(in_obj)
      cursor.execute("INSERT INTO augmented_objectives (input_objective, audience_score, behavior_score, condition_score, degree_score, new_objective) VALUES (?, ?, ?, ?, ?, ?)", (json.dumps(in_obj), row[2], row[3], row[4], row[5], json.dumps(new_obj)))

  conn.commit()
  conn.close()

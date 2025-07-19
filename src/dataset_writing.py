from pathlib import Path
import random, csv, json, itertools
from typing import List, Callable, Tuple
from num2words import num2words
import ast
import pandas as pd


N_ROWS = 100 # change to 1,000,000 later. right now with ages that make sense, 100 rows seems best for a small experiment. FOR NOW.
AGE_RANGE = range(0, 100)

"""def generate_sentence(n_rows = N_ROWS, ages = AGE_RANGE, seed=42):
    rng = random.Random(seed)
    # not really a need for it to be a csv, could just be a .txt and its fine. closer to what we want, even.
    with open("data/ages.txt", "w", newline="") as f:
        #writer = csv.writer(f)
        for i in range(n_rows):
            age = random.choice(ages)
            #writer.writerow([f"My age is {age}"])
            f.write(f"My age is {age}\n")
"""

#generate_sentence()
#print("Text written!")

def load_sentences(path: Path) -> List[str]:
    sentences = []
    with open(path, "r") as f:
        for line in f:
            sentences.append(line.strip())

    print("Number of sentences:", len(sentences))
    print(sentences[:10])
    return sentences

def generate_sentences(out_path, n_rows=N_ROWS, ages=AGE_RANGE, seed=42, style="digits"):
    """
    style:
    "digits" -> 38
    "float" -> 38.00
    "scientific" -> 3.80e+01
    """
    rng = random.Random(seed)

    if style == "digits":
        age_format: Callable[[int], str] = lambda age: f"{age}"
    elif style == "float":
        age_format = lambda age: f"{age:.2f}"
    elif style == "scientific":
        age_format = lambda age: f"{age:.2e}"
    elif style == "words":
        age_format = lambda age: num2words(age)
    else:
        raise ValueError(f"Unrecognized style: {style}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        for i in range(n_rows):
            age = rng.choice(ages)
            rep = age_format(age)
            sentence = f"My age is {rep}"
            f.write(sentence + "\n")

    print("file written to: ", out_path)


def clean_raw_data():
    RAW_PATH = Path("../data/data.csv")
    CLEANED_PATH = Path("../data/cleaned_data.csv")

    df_raw = pd.read_csv(RAW_PATH, header=0, index_col=0, names=["attributes_raw"])

    df_raw["attributes"] = df_raw["attributes_raw"].apply(ast.literal_eval)
    df = pd.DataFrame(df_raw["attributes"].tolist())


    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Wrote cleaned dataset to {CLEANED_PATH}")

#clean_raw_data()


def load_bigger_dataset(path: Path) -> List[str]:
    sentences = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            sentences.append(row[7])

    print("Number of sentences:", len(sentences))
    print(sentences[:10])
    return sentences
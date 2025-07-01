from pathlib import Path
import random, csv, json, itertools

N_ROWS = 100 # change to 1,000,000 later. right now with ages that make sense, 100 rows seems best for a small experiment. FOR NOW.
AGE_RANGE = range(0, 100)

def generate_sentence(n_rows = N_ROWS, ages = AGE_RANGE, seed=42):
    rng = random.Random(seed)
    # not really a need for it to be a csv, could just be a .txt and its fine. closer to what we want, even.
    with open("data/ages.txt", "w", newline="") as f:
        #writer = csv.writer(f)
        for i in range(n_rows):
            age = random.choice(ages)
            #writer.writerow([f"My age is {age}"])
            f.write(f"My age is {age}\n")


generate_sentence()
print("Text written!")
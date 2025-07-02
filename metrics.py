import numpy as np
from typing import List, Tuple

def failproof(query_age: int, sentences):
    ages = [int(s.split()[-1]) for s in sentences]
    differences = np.abs(np.array(ages) - query_age)
    index = int(np.argmin(differences))
    return ages[index], index

def compute_errors(true: List[int], predicted: List[int], acceptable_range=2):
    errors = [abs(t-p) for t,p in zip(true, predicted)]
    correct = [e <= acceptable_range for e in errors]
    accuracy = sum(correct) / len(true)
    return errors, correct, accuracy
import numpy as np
from typing import List, Tuple
import re
from word2number import w2n


#def _parse_age(sentence: str) -> int:
"""
Grab the first integer if present, else convert to int.
Remember you're using .lower() for normalizing / trying out bert-cased vs uncased, becomes not too useful
:param sentence:
:return:
"""
"""m = re.search(r"\b(\d+)\b", sentence)
if m:
    return int(m.group(1))

txt = sentence.lower().split("my age is")[-1]
txt = re.sub(r"[^a-z\s-]", " ", txt).strip()
return w2n.word_to_num(txt)"""

def _parse_age(sentence: str) -> float:
    m = re.findall(r"\d+(?:\.\d+)?", sentence)
    if not m:
        raise ValueError(f"No number found in: {sentence}")
    return float(m[-1])

def failproof(query_age: int, sentences):
    #ages = [int(s.split()[-1]) for s in sentences]

    #ages = [_parse_age(s) for s in sentences]
    nums = np.array([_parse_age(s) for s in sentences], dtype=float)
    #differences = np.abs(np.array(ages) - query_age)
    #index = int(np.argmin(differences))
    index = int(np.abs(nums - query_age).argmin())
    #return ages[index], index
    return float(nums[index]), index

def compute_errors(true: List[int], predicted: List[int], acceptable_range=2):
    signed_errors = [(t-p) for t,p in zip(true, predicted)]
    errors = [abs(e) for e in signed_errors]
    correct = [e <= acceptable_range for e in errors]
    exact = [e == 0 for e in errors]
    accuracy = sum(correct) / len(true)
    exact_match_accuracy = sum(exact) / len(true)


    return errors, correct, accuracy, signed_errors, exact_match_accuracy
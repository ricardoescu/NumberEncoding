import numpy as np
from typing import List, Tuple
import re
from word2number import w2n


def _parse_age(sentence: str) -> int:
    """
    Grab the first integer if present, else convert to int.
    Remember you're using .lower() for normalizing / trying out bert-cased vs uncased, becomes not too useful
    :param sentence:
    :return:
    """
    m = re.search(r"\b(\d+)\b", sentence)
    if m:
        return int(m.group(1))

    txt = sentence.lower().split("my age is")[-1]
    txt = re.sub(r"[^a-z\s-]", " ", txt).strip()
    return w2n.word_to_num(txt)

def failproof(query_age: int, sentences):
    #ages = [int(s.split()[-1]) for s in sentences]

    ages = [_parse_age(s) for s in sentences]
    differences = np.abs(np.array(ages) - query_age)
    index = int(np.argmin(differences))
    return ages[index], index

def compute_errors(true: List[int], predicted: List[int], acceptable_range=2):
    signed_errors = [(t-p) for t,p in zip(true, predicted)]
    errors = [abs(e) for e in signed_errors]
    correct = [e <= acceptable_range for e in errors]
    exact = [e == 0 for e in errors]
    accuracy = sum(correct) / len(true)
    exact_match_accuracy = sum(exact) / len(true)


    return errors, correct, accuracy, signed_errors, exact_match_accuracy
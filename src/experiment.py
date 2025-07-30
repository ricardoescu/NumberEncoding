
from metrics import _parse_age, failproof, compute_errors
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from itertools import islice
import csv
import random
from models import SentenceEncoder
from num2words import num2words

def run_experiment(encoder, sentences, ages: range, word_query: bool = False) -> dict:

    true_ages, pred_ages = [], []
    corpus_embedding = encoder.embed(sentences)
    for q in ages:
        true, true_index = failproof(q, sentences)
        if word_query:
            q = num2words(q)
        model_index = encoder.find_best_index(q, corpus_embedding)
        pred = _parse_age(sentences[model_index])

        true_ages.append(true)
        pred_ages.append(pred)

    errors, correct, accuracy, non_abs_errors, exact_match_accuracy = compute_errors(true_ages, pred_ages)

    results = {
        "true": true_ages,
        "pred": pred_ages,
        "errors": errors,
        "correct": correct,
        "accuracy": accuracy,
        "full_error": non_abs_errors,
        "exact_match_acc": exact_match_accuracy
    }

    return results


def get_column_choices(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Reads the header + first N rows of the csv and returns:
        - header (column names)
        - choices: Potential choices present in the csv in order to be able to switch one of them.
    :param path:
    :return:
    """
    choices: Dict[str, set] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # initialize the sets
        for col in header:
            choices[col] = set()

        # sample first n rows (I will leave at 100 for now, as my computer has trouble processing all 16k)
        for row in islice(reader, 100):
            for col, val in zip(header, row):
                if val: choices[col].add(val)
    return header, {col: sorted(vals) for col, vals in choices.items()}

def edit_column(sentence: str, header: List[str], choices: Dict[str, List[str]], delimiter: str = ", the ") -> str:
    """
    Given one sentence built with all columns in "header",
    with 'delimiter' pick one column at random and replaces its value with
    a different random choice from what appears in the dataset.

    ALL columns have the changing info after an "is", so that allows to identify the choices
    (although, a column switch with another row could be another way of doing the same)
    :param sentence:
    :param header:
    :param choices:
    :param delimiter:
    :return:
    """
    raw = sentence.split(delimiter)

    parts = [raw[0]] + [f"the {p}" for p in raw[1:]]

    col_idx = random.randrange(len(parts))
    col_name = header[col_idx]

    # current value (everything after the "is ")
    prefix, sep, current = parts[col_idx].partition(" is ")
    options = choices.get(col_name, [])
    #if not options:
     #   return sentence # nothing can be done

    # remove current and pick a random one
    candidates = [v for v in options if v != current]
    #if not candidates:
     #   return sentence

    new_val = random.choice(candidates)
    parts[col_idx] = new_val
    return delimiter.join([parts[0]] + [p[len("the "):] for p in parts[1:]])

def run_experiment_full(encoder: SentenceEncoder, csv_path: Path, sentences: List[str], max_queries: Optional[int] = None) -> Dict[str, List]:
    header, choices = get_column_choices(csv_path)


    gold, pred, correct, pred_sents = [], [], [], []
    N = len(sentences) if max_queries is None else min(len(sentences), max_queries)
    corpus_embedding = encoder.embed(sentences[:N])
    queries = []

    for i in range(N):
        gold.append(i)
        query = edit_column(sentences[i], header, choices)
        queries.append(query)
        p = encoder.find_best_index(query, corpus_embedding)
        pred.append(p)
        correct.append(p==i)
        pred_sents.append(sentences[p])

    accuracy = sum(correct) / len(correct)
    return {
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "queries": queries,
        "pred_sents": pred_sents,
        "accuracy": accuracy,
    }
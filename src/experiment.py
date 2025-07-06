from typing import List
from pathlib import Path
import numpy as np

from dataset_writing import load_sentences, generate_sentence
from models import SentenceEncoder
from metrics import failproof, compute_errors

def run_experiment(encoder, sentences, ages: range) -> dict:

    true_ages, pred_ages = [], []
    for q in ages:
        model_index = encoder.find_best_index(q, sentences)
        pred = int(sentences[model_index].split()[-1])
        true, true_index = failproof(q, sentences)

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
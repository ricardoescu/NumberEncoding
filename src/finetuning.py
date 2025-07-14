import logging, pandas as pd, itertools
from sentence_transformers import SentenceTransformer, models, losses, InputExample, LoggingHandler
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import DatasetInfo
from collections import defaultdict
from metrics import _parse_age, failproof

from datasets import DatasetInfo
DatasetInfo.dataset_name = property(lambda self: self.builder_name)

df = pd.read_csv("../data/cleaned_data.csv")
lines = df.iloc[:,0].astype(str).tolist()

ages = [_parse_age(s) for s in lines]
idx_by_age = list(enumerate(ages))

# for each anchor pick:
# a positive = sentence with minimal non-zero |age_diff|
# a negative = sentence with maximal |age_diff|
triplets = []
for i, age in idx_by_age:
    # find all other sentences0 (index, |diff)
    diffs = [(j, abs(age-other)) for j, other in idx_by_age if j!= i]
    # Pick nearest and farthest
    positive_i, _ = min(diffs, key=lambda x: x[1])
    negative_i, _ = max(diffs, key=lambda x: x[1])
    triplets.append(
        InputExample(texts=[lines[i], lines[positive_i], lines[negative_i]])
    )

loader = DataLoader(triplets, shuffle=True, batch_size=32)
model = SentenceTransformer("bert-base-nli-mean-tokens")
loss_fn = losses.TripletLoss(model)

logging.basicConfig(level=logging.INFO, handlers=[LoggingHandler()])
model.fit(train_objectives=[(loader, loss_fn)], epochs=2, warmup_steps=100, show_progress_bar=True)

model.save("nli-mean-numeric-triplet")
print("Saved to nli-mean-numeric-triplet")
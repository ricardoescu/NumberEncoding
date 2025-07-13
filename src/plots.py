import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple
from metrics import failproof, _parse_age
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_tsne(model_name, out_folder, query: int, sentences: List[str], encoder, pca_dims: int, perp: int, tsne_dimensions:int = 2):
    query_text = f"My age is {query}"
    numbers = list(range(100))
    s_numbers = [str(i) for i in numbers]
    emb_nums = encoder.embed(s_numbers)

    emb_query = encoder.embed([str(query)][0])
    #emb_query = encoder.embed(query_text)
    num_sentence = [str(_parse_age(s)) for s in sentences]
    emb_sentences = encoder.embed(sentences)
    #emb_sentences = encoder.embed(num_sentence)

    model_index = encoder.find_best_index(query, emb_sentences)
    true_age, true_index = failproof(query, sentences)


    all_embeddings = np.vstack([emb_query, emb_sentences, emb_sentences[true_index: true_index+1], emb_nums])

    # Apply dimensionality reduction (optional, as it does not take that much time)
    # Important to note! Models such as RoBERTa did take long. Gotta look more into if this actually makes it considerably better
    # Dimensionality reduction did not affect the graphing much.
    pca = PCA(n_components=pca_dims, random_state=42)
    pca_transformed = pca.fit_transform(all_embeddings)

    # Ensuring that perplexity is always within the limits of the shape.
    perplexity = min(perp, pca_transformed.shape[0]-1)
    #perplexity = min(perp, all_embeddings.shape[0] - 1)

    tsne = TSNE(n_components=tsne_dimensions, perplexity=perplexity, random_state=42) # I think specifying the random state helps for replicability? I need to check again.
    coords = tsne.fit_transform(pca_transformed)
    #coords = tsne.fit_transform(all_embeddings)

    query_coord = coords[0]
    sentence_coords = coords[1:1 + len(sentences)]
    best_coords = coords[1 + len(sentences)]
    all_num_coords = coords[1+len(sentences)+1:]

    ages = [_parse_age(s) for s in sentences]

    plt.figure(figsize=(6, 6))
    # plot all sentences

    sc1 = plt.scatter(sentence_coords[:, 0], sentence_coords[:, 1], c=ages, cmap='Blues', s=30, alpha=0.8, label='Sentences')
    plt.colorbar(sc1, label='sentence age')

    sc2 = plt.scatter(all_num_coords[:, 0], all_num_coords[:, 1], c=numbers, cmap='OrRd', s=20, alpha=0.6, marker=".", label='Numbers')
    plt.colorbar(sc2, label='Number values')

    plt.scatter(*query_coord, marker='^', s=100, color='black', label=f"query={query}")
    plt.scatter(*sentence_coords[model_index], marker='X', s=120, color='red', label=f"model best={sentences[model_index].split()[-1]}")
    plt.scatter(*best_coords, marker='s', s=120, color='green', label=f"true best={true_age}")
    plt.legend()

    plt.title(f"{model_name} t-SNE projection")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()

    filename = f"tsne_{model_name}_{query}.png"
    plt.savefig(out_folder / filename)

    plt.show()

def plot_concentration(model_name, out_folder, ages, preds, errors):
    plt.figure(figsize=(6,6))
    sc = plt.scatter(ages, preds, c=errors, cmap='coolwarm', s=30)
    plt.plot([min(ages), max(ages)], [min(ages), max(ages)])
    plt.colorbar(sc, label='|pred - true|')
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title(f"{model_name}: Preds vs True ages")

    filename = f"concentration_{model_name}.png"
    plt.savefig(out_folder / filename)

    plt.show()


def heatmap(results, model_name, out_folder):
    df = pd.DataFrame({
        "true": results["true"],
        "pred": results["pred"]
    })

    # Define bins.
    # Grouped by 5 or 10 years gives great interpretability,
    # but can also be good for visualising +-2 years, or as required.
    year_bins = 10
    bins = list(range(0, 101, year_bins))
    labels = [f"{i:02d}–{i + 9:02d}" for i in bins[:-1]]  # ["00–09", "10–19", …, "90–99"]

    # Bin both true and pred
    df["true_bin"] = pd.cut(df.true, bins=bins, labels=labels, right=False)
    df["pred_bin"] = pd.cut(df.pred, bins=bins, labels=labels, right=False)

    # Pivot to get counts per (true_bin × pred_bin)
    heat = df.pivot_table(
        index="true_bin",
        columns="pred_bin",
        aggfunc="size",
        fill_value=0
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heat,
        annot=True,  # show counts in each cell
        fmt="d",  # integer format
        cmap="Blues",
        cbar_kws={"label": "Count"}
    )
    plt.xlabel("Predicted age bin")
    plt.ylabel("True age bin")
    plt.title(f"{model_name}: True vs Predicted Age (10-year bins)")
    plt.tight_layout()

    filename = f"heatmap_{model_name}_{year_bins}.png"
    plt.savefig(out_folder / filename)

    plt.show()


def plot_concentration_color_gradient(results, model_name, out_folder):
    """
    Scatter plot: colour encodes the *true* age (0‒99) on a single gradient,
    marker size encodes absolute error.
    """

    cmap = "plasma"
    figsize = (6, 6)

    true = results["true"]
    preds = results["pred"]
    errors = results["errors"]

    true_arr = np.array(true)
    pred_arr = np.array(preds)
    err_arr = np.array(errors)

    # Changing sizes with how big the error is for better representation while maintaining the color gradient asked for.
    sizes = 30 + err_arr * 10  # 0 error -> size 30, 10 error -> size 130

    plt.figure(figsize=figsize)
    sc = plt.scatter(true_arr, pred_arr,
                     c=true_arr,  # colour = true age
                     s=sizes,
                     cmap=cmap,
                     alpha=0.8,
                     edgecolors="none")

    # perfect-prediction line
    # not entirely sure if this is necessary. Might remove.
    plt.plot([0, 99], [0, 99], linestyle="--", color="grey", linewidth=1)

    # colorbar shows age scale
    cbar = plt.colorbar(sc)
    cbar.set_label("True age")

    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title("Age-prediction dispersion (colour = true age, size = error)")
    plt.tight_layout()

    filename = f"concentration-color_gradient{model_name}.png"
    plt.savefig(out_folder / filename)

    plt.show()

def plot_prediction_histogram(pred_ages, bins=range(0, 101, 5), model_name="", out_folder=""):
    """
    Histogram of predicted ages.
    """
    plt.figure(figsize=(8,4))
    plt.hist(pred_ages, bins=bins, edgecolor="black")
    plt.xlabel("Predicted age")
    plt.ylabel("Count")
    plt.title("Distribution of predicted ages")
    plt.tight_layout()

    filename = f"{model_name}_prediction_histogram.png"
    plt.savefig(out_folder / filename)

    plt.show()

def plot_error_histogram(errors, model_name, out_dir, absolute=True):
    """
    Histogram of absolute errors |pred–true|.
    Modify to total error?
    """

    bins = range(min(errors), max(errors)+2, 10)
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=bins, edgecolor="black", align="left")
    if absolute:
        xlab = "Absolute Error"
    else:
        xlab = "Error"
    plt.xlabel(xlab)
    plt.ylabel("Count")
    plt.title("Error distribution")
    plt.xticks(bins)
    plt.tight_layout()

    filename = f"{model_name}_histogram_{xlab}.png"
    plt.savefig(out_dir / filename)

    plt.show()


def results_summary(DATASETS, accuracy="accuracy"):

    #DATASETS = {"digits", "float", "scientific", "words"}
    #DATASETS = {"bigger_dataset"}
    root = Path("../results")

    VALID_MODELS = {
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "bert-base-nli-mean-tokens",
        "bert-base-uncased",
        "bert-base-cased",
        "all-roberta-large-v1",
        "all-MiniLM-12-v2",
        "intfloat/e5-small",
        "princeton-nlp/unsup-simcse-bert-base-uncased",
        "mathbert-base-uncased",
    }

    records = []

    for f in root.rglob("*_summary.csv"):
        # find the first parent folder that matches a known dataset
        dataset_folder = next((p for p in f.parents if p.name in DATASETS), None)
        if dataset_folder is None:
            continue

        dataset = dataset_folder.name
        # everything *below* the dataset folder is the (possibly nested) model path
        model_tag = f.parent.relative_to(dataset_folder).as_posix()  # keeps slashes

        if VALID_MODELS and model_tag not in VALID_MODELS:
            continue # ignore models outside the list

        acc = pd.read_csv(f).loc[0, accuracy]
        records.append({"model": model_tag, "dataset": dataset, "accuracy": acc})

    overall = (pd.DataFrame(records)
               .pivot_table(index="model",
                            columns="dataset",
                            values="accuracy",
                            aggfunc="first")
               .sort_index())

    print(overall.to_markdown(floatfmt=".3f"))
    overall.to_csv(root / "overall_summary.csv")
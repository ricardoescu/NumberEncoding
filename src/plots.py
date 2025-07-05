import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple
from metrics import failproof

def plot_tsne(query: int, sentences: List[str], encoder, pca_dims: int, perp: int, tsne_dimensions:int = 2):
    emb_query = encoder.embed([str(query)][0])
    emb_sentences = encoder.embed(sentences)

    model_index = encoder.find_best_index(query, sentences)
    true_age, true_index = failproof(query, sentences)


    all_embeddings = np.vstack([emb_query, emb_sentences, emb_sentences[true_index: true_index+1]])

    # Apply dimensionality reduction (optional, as it does not take that much time)
    # Important to note! Models such as RoBERTa did take long. Gotta look more into if this actually makes it considerably better
    # Dimensionality reduction did not affect the graphing much.
    pca = PCA(n_components=pca_dims, random_state=42)
    pca_transformed = pca.fit_transform(all_embeddings)

    # Ensuring that perplexity is always within the limits of the shape.
    perplexity = min(perp, pca_transformed.shape[0]-1)

    tsne = TSNE(n_components=tsne_dimensions, perplexity=perplexity, random_state=42) # I think specifying the random state helps for replicability? I need to check again.
    coords = tsne.fit_transform(all_embeddings)

    query_coord = coords[0]
    sentence_coords = coords[1:]
    best_coords = coords[-1]

    plt.figure(figsize=(6, 6))
    # plot all sentences

    plt.scatter(sentence_coords[:, 0], sentence_coords[:, 1], marker='o', alpha=0.6)
    plt.scatter(*query_coord, marker='^', s=100, label=f"query={query}")
    plt.scatter(*sentence_coords[model_index], marker='X', s=120, label=f"model best={sentences[model_index].split()[-1]}")
    plt.scatter(*best_coords, marker='s', s=120, label=f"true best={true_age}")
    plt.legend()

    plt.title(f"t-SNE projection")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    plt.show()

def plot_concentration(ages, preds, errors):
    plt.figure(figsize=(6,6))
    sc = plt.scatter(ages, preds, c=errors, cmap='coolwarm', s=30)
    plt.plot([min(ages), max(ages)], [min(ages), max(ages)])
    plt.colorbar(sc, label='|pred - true|')
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title("Predicted vs True ages")
    plt.show()
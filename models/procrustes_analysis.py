import fasttext

import numpy as np

import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
import sklearn.manifold
import ot


def orthogonal_procrustes(X: "fasttext.FastText", Y: "fasttext.FastText"):
    """
    Align two sets of embeddings (X and Y) using orthogonal_procrustes.

    Parameters:
    - X: Source embeddings (numpy array of shape (n_samples, n_features))
    - Y: Target embeddings (numpy array of shape (n_samples, n_features))

    Returns:
    - Y_aligned: Aligned target embeddings to the source embeddings space
    """
    common_vocab = set(X.words).intersection(set(Y.words))

    # Extract word vectors for the common vocabulary
    X_vectors = np.array([X.get_word_vector(word) for word in common_vocab])
    Y_vectors = np.array([Y.get_word_vector(word) for word in common_vocab])
    R, scale = orthogonal_procrustes(Y_vectors, X_vectors)
    Y_aligned = Y_vectors.dot(R)
    norms = np.linalg.norm(Y_aligned, axis=1, keepdims=True)
    normalized_aligned_matrix = Y_aligned / norms
    return normalized_aligned_matrix


def Wasserstein_Procrustes(X, Y):
    """
    Align two sets of embeddings (X and Y) using an approach inspired by Wasserstein Procrustes.

    Parameters:
    - X: Source embeddings (numpy array of shape (n_samples, n_features))
    - Y: Target embeddings (numpy array of shape (n_samples, n_features))

    Returns:
    - Y_aligned: Aligned target embeddings to the source embeddings space
    """
    # Normalize embeddings to unit norm
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # Compute the cost matrix as the negative cosine similarity or Euclidean distance
    M = ot.dist(X_normalized, Y_normalized, metric="euclidean")

    # Compute the optimal transport plan using the Sinkhorn algorithm
    transport_plan = ot.sinkhorn2(
        np.ones((X.shape[0],)), np.ones((Y.shape[0],)), M, 1e-1
    )[0]

    # Map Y to X space using the transport plan
    Y_mapped = transport_plan.dot(Y_normalized)

    # Compute the optimal orthogonal transformation (SVD)
    U, _, Vt = np.linalg.svd(Y_mapped.T.dot(X_normalized), full_matrices=False)
    Q = U.dot(Vt)  # Optimal rotation matrix

    # Apply the transformation to Y
    Y_aligned = Y_normalized.dot(Q)

    return Y_aligned

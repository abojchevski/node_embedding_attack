"""
Implementation of the method proposed in the paper:

'Adversarial Attacks on Node Embeddings via Graph Poisoning'
Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
http://proceedings.mlr.press/v97/bojchevski19a.html

Copyright (C) owned by the authors, 2019
"""

import numba
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec


def deepwalk_skipgram(adj_matrix, embedding_dim=64, walk_length=80, walks_per_node=10,
                      workers=8, window_size=10, num_neg_samples=1):
    """Compute DeepWalk embeddings for the given graph using the skip-gram formulation.

    Parameters
    ----------
    adj_matrix : sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    embedding_dim : int, optional
        Dimension of the embedding
    walks_per_node : int, optional
        Number of walks sampled from each node
    walk_length : int, optional
        Length of each random walk
    workers : int, optional
        Number of threads (see gensim.models.Word2Vec process)
    window_size : int, optional
        Window size (see gensim.models.Word2Vec)
    num_neg_samples : int, optional
        Number of negative samples (see gensim.models.Word2Vec)

    Returns
    -------
    E : np.ndarray, shape [num_nodes, embedding_dim]
        Embedding matrix

    """
    walks = sample_random_walks(adj_matrix, walk_length, walks_per_node)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=1, negative=num_neg_samples, hs=0, compute_loss=True)
    embedding = model.wv.syn0[np.fromiter(map(int, model.wv.index2word), np.int32).argsort()]
    return embedding


def deepwalk_svd(adj_matrix, window_size, embedding_dim, num_neg_samples=1, sparse=True):
    """Compute DeepWalk embeddings for the given graph using the matrix factorization formulation.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param window_size: int
        Size of the window
    :param embedding_dim: int
        Size of the embedding
    :param num_neg_samples: int
        Number of negative samples
    :param sparse: bool
        Whether to perform sparse operations
    :return: np.ndarray, shape [num_nodes, embedding_dim]
        Embedding matrix.
    """
    sum_powers_transition = sum_of_powers_of_transition_matrix(adj_matrix, window_size)

    deg = adj_matrix.sum(1).A1
    deg[deg == 0] = 1
    deg_matrix = sp.diags(1 / deg)

    volume = adj_matrix.sum()

    M = sum_powers_transition.dot(deg_matrix) * volume / (num_neg_samples * window_size)

    log_M = M.copy()
    log_M[M > 1] = np.log(log_M[M > 1])
    log_M = log_M.multiply(M > 1)

    if not sparse:
        log_M = log_M.toarray()

    Fu, Fv = svd_embedding(log_M, embedding_dim, sparse)

    loss = np.linalg.norm(Fu.dot(Fv.T) - log_M, ord='fro')

    return Fu, Fv, loss, log_M


def sample_random_walks(adj_matrix, walk_length, walks_per_node, seed=None):
    """Sample random walks of fixed length from each node in the graph in parallel.

    Parameters
    ----------
    adj_matrix : sp.csr_matrix, shape [n_nodes, n_nodes]
        Sparse adjacency matrix
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    seed : int or None
        Random seed

    Returns
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        The sampled random walks

    """
    if seed is None:
        seed = np.random.randint(0, 100000)
    adj_matrix = sp.csr_matrix(adj_matrix)
    random_walks = _random_walk(adj_matrix.indptr,
                                adj_matrix.indices,
                                walk_length,
                                walks_per_node,
                                seed).reshape([-1, walk_length])
    return random_walks


@numba.jit(nopython=True)
def _random_walk(indptr, indices, walk_length, walks_per_node, seed):
    """Sample r random walks of length l per node in parallel from the graph.

    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    walk_length : int
        Random walk length
    walks_per_node : int
        Number of random walks per node
    seed : int
        Random seed

    Returns
    -------
    walks : array-like, shape [r*N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for ir in range(walks_per_node):
        for n in range(N):
            for il in range(walk_length):
                walks.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])

    return np.array(walks)


def svd_embedding(x, embedding_dim, sparse=False):
    """Computes an embedding by selection the top (embedding_dim) largest singular-values/vectors.

    :param x: sp.csr_matrix or np.ndarray
        The matrix that we want to embed
    :param embedding_dim: int
        Dimension of the embedding
    :param sparse: bool
        Whether to perform sparse operations
    :return: np.ndarray, shape [?, embedding_dim], np.ndarray, shape [?, embedding_dim]
        Embedding matrices.
    """
    if sparse:
        U, s, V = sp.linalg.svds(x, embedding_dim)
    else:
        U, s, V = np.linalg.svd(x)

    S = np.diag(s)
    Fu = U.dot(np.sqrt(S))[:, :embedding_dim]
    Fv = np.sqrt(S).dot(V)[:embedding_dim, :].T

    return Fu, Fv


def sum_of_powers_of_transition_matrix(adj_matrix, pow):
    """Computes \sum_{r=1}^{pow) (D^{-1}A)^r.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param pow: int
        Power exponent
    :return: sp.csr_matrix
        Sum of powers of the transition matrix of a graph.

    """
    deg = adj_matrix.sum(1).A1
    deg[deg == 0] = 1
    transition_matrix = sp.diags(1 / deg).dot(adj_matrix)

    sum_of_powers = transition_matrix
    last = transition_matrix
    for i in range(1, pow):
        last = last.dot(transition_matrix)
        sum_of_powers += last

    return sum_of_powers

import numpy as np
import networkx as nx
from scipy import sparse
from utils import load_data_into_graph
import logging

"""
Parameters:
    - Window size
    - Negative sample size
    - Embedding size
    - Graph
"""


class NetMF:
    def __init__(self, G: nx.graph, window: int, neg_samples: int, embedding_size: int, logger=logging.getLogger("NetMF")):
        self.A = nx.to_numpy_array(G)  # adjacency matrix
        self.volume = np.sum(self.A)
        self.window = window
        self.neg_samples = neg_samples
        self.embedding_size = embedding_size
        self.n = self.A.shape[0]
        self.logger = logger

    # Directly computing M
    def small_window_size(self):
        self.logger.info("Computing for small window size")
        D = np.sum(self.A, axis=0)
        D_inv = np.diag(D ** -1)

        P = D_inv @ self.A

        S = np.zeros_like(P)
        P_pow = np.identity(self.n)

        for r in range(self.window):
            P_pow = P_pow @ P
            S += P_pow

        S *= self.volume / (self.neg_samples * self.window)
        M = S @ D_inv

        # M'
        M = sparse.csr_array(np.maximum(M, 1))
        self.logger.info("Almost done...")

        return self.svd(M)

    def large_window_size(self, rank):
        self.logger.info("Computing for large window size")
        laplacian, D = sparse.csgraph.laplacian(
            self.A, return_diag=True, normed=True)
        D_inv_sqrt = sparse.diags(D ** -0.5)
        # D^-1/2 A D^-1/2
        D_A_D = sparse.identity(self.n) - laplacian

        self.logger.info("Laplacian computed")

        # Approximating it with first `rank eigenvalues
        # Gives you eigenvals in increasing order tho
        # LA means it returns first `rank` largest eigenvalues
        eigenvals, eigenvecs = sparse.linalg.eigsh(D_A_D, rank, which='LA')

        self.logger.info("Eigenvectors computed")

        eigenval_power_sum = np.zeros_like(eigenvals)
        # First time it's just gonna give us eigenvals as the result
        current_power = np.ones_like(eigenvals)
        for _ in range(self.window):
            current_power = current_power * eigenvals
            eigenval_power_sum += current_power

        self.logger.info("Power computed")

        D_inv_sqrt_U = D_inv_sqrt.dot(eigenvecs)
        normalization = self.volume / (self.neg_samples * self.window)

        result = normalization * D_inv_sqrt_U * eigenval_power_sum @ D_inv_sqrt_U.T
        # M'
        M = np.maximum(result, 1)

        self.logger.info("Almost done...")
        return self.svd(M)

    def svd(self, M):
        # TODO: lower precision of floats?
        # SVD
        u, s, _ = sparse.linalg.svds(
            M, self.embedding_size, return_singular_vectors='u')
        # U_D
        return u @ sparse.diags(np.sqrt(s))


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp
    G, _, _, _ = load_data_into_graph('BlogCatalog')
    logger = logging.getLogger(__name__)

    netmf = NetMF(G, 10, 1, 128, logger=logger)

    # rank is selected from sec 4 in the paper.
    embedding = netmf.large_window_size(256)
    np.save("models/NetMF-BlogCatalog.npy", embedding, allow_pickle=False)


if __name__ == "__main__":
    main()

import numpy as np
import networkx as nx
from scipy import sparse
from utils import load_data_into_graph
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from color_communities import color_communities

"""
Parameters:
    - Window size
    - Negative sample size
    - Embedding size
    - Graph
"""


class NetMF:
    def __init__(self, G: nx.graph, window: int, neg_samples: int, embedding_size: int, logger=logging.getLogger("NetMF")):
        self.A = nx.to_numpy_array(
            G, nodelist=np.sort(G.nodes()))  # adjacency matrix
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
        M = np.maximum(M, 1)
        self.logger.info("Almost done...")

        return self.svd(np.log(M))

    def large_window_size(self, rank):
        self.logger.info("Computing for large window size")
        laplacian, D = sparse.csgraph.laplacian(
            self.A, return_diag=True, normed=True)
        # NOTE: this comes from the authors
        D_inv_sqrt = sparse.diags(D ** -1.0)
        # D^-1/2 A D^-1/2
        D_A_D = sparse.identity(self.n) - laplacian

        self.logger.info("Laplacian computed")

        # Approximating it with first `rank eigenvalues
        # Gives you eigenvals in increasing order tho
        # LA means it returns first `rank` largest eigenvalues
        eigenvals, eigenvecs = sparse.linalg.eigsh(D_A_D, rank, which='LA')
        # NOTE: this comes from the authors' implementation for no good reason at all
        eigenvals = self.deepwalk_filter(eigenvals)

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
        return self.svd(np.log(M))

    def deepwalk_filter(self, evals):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1. if x >= 1 else x * \
                (1-x**self.window) / (1-x) / self.window
        evals = np.maximum(evals, 0)
        self.logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
                         np.max(evals), np.min(evals))
        return evals

    def svd(self, M):
        # TODO: lower precision of floats?
        # SVD
        u, s, _ = sparse.linalg.svds(
            M, k=self.embedding_size, return_singular_vectors='u')
        # U_D
        # NOTE: another author's way of doing it
        return sparse.diags(np.sqrt(s)).dot(u.T).T


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp
    logger = logging.getLogger(__name__)

    # G = nx.karate_club_graph()
    # netmf = NetMF(G, 5, 1, 32, logger=logger)
    # embedding = netmf.small_window_size()
    # x_pca = PCA(n_components=2).fit_transform(embedding)
    # colors = color_communities(G)
    # ax = plt.subplot(1, 1, 1)
    # ax.scatter(x_pca[:, 0], x_pca[:, 1], c=colors)
    # for node in G.nodes():
    #     ax.annotate(
    #         node, (x_pca[node, 0], x_pca[node, 1]))
    # plt.show()

    G, _, _, _ = load_data_into_graph('Flickr')

    netmf = NetMF(G, 10, 1, 128, logger=logger)

    # rank is selected from sec 4 in the paper.
    embedding = netmf.large_window_size(16384)
    np.save("models/NetMF-Flickr-their-ordered.npy",
            embedding, allow_pickle=False)


if __name__ == "__main__":
    main()

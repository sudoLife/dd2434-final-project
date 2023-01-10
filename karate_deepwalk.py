from DeepWalk import DeepWalk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from color_communities import color_communities


def main():
    G = nx.karate_club_graph()

    embedding = 10

    deepwalk = DeepWalk(G, window=8, embedding_size=embedding,
                        walks_per_vertex=80, walk_length=20)
    corpus = deepwalk.generate_corpus()
    embedded = deepwalk.train(corpus, workers=1, epochs=1)

    reduced_embedding = PCA(n_components=2).fit_transform(embedded)

    colors = color_communities(G)

    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=14)
    nx.draw(G, pos, with_labels=True, node_color=colors)

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=colors)

    for node in G.nodes():
        ax2.annotate(
            node, (reduced_embedding[node, 0], reduced_embedding[node, 1]))

    plt.show()


if __name__ == "__main__":
    main()

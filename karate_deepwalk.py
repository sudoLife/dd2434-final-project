from DeepWalk import DeepWalk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def main():
    G = nx.karate_club_graph()

    # Use the FastGreedy algorithm to identify communities in the network
    communities = list(
        nx.algorithms.community.greedy_modularity_communities(G))

    # Print the communities
    print(f'Number of communities: {len(communities)}')
    for i, community in enumerate(communities):
        print(f'Community {i+1}: {community}')

    # Create a mapping from nodes to community IDs
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    embedding = 10

    deepwalk = DeepWalk(G, window=8, embedding=embedding,
                        walksPerVertex=80, walkLength=20, epochs=1)
    corpus = deepwalk.generate_corpus()
    model = deepwalk.train(corpus, workers=1)

    embedded = np.zeros((len(G), embedding))

    colors = [node_to_community[n] * 0.5 for n in G.nodes()]

    for node in G.nodes():
        embedded[node] = model.wv[str(node)]

    reducedEmbedding = PCA(n_components=2).fit_transform(embedded)

    ax = plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=14)
    nx.draw(G, pos, with_labels=True, node_color=[
            node_to_community[n] for n in G.nodes])

    ax = plt.subplot(1, 2, 2)
    ax.scatter(reducedEmbedding[:, 0], reducedEmbedding[:, 1], c=colors)

    for node in G.nodes():
        ax.annotate(
            node, (reducedEmbedding[node, 0], reducedEmbedding[node, 1]))

    plt.show()


if __name__ == "__main__":
    main()

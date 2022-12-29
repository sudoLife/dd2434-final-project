import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from Node2Vec import Node2Vec
from color_communities import color_communities


def read_graph(input_edgelist, weighted=False, directed=False):
    """
    Reads the input network in networkx.
    """
    if weighted:
        G = nx.read_edgelist(input_edgelist, nodetype=int, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_edgelist, nodetype=int,
                             create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not directed:
        G = G.to_undirected()

    return G


def main():
    output_file = 'models/karate_node2vec.kv'

    graph = nx.karate_club_graph()

    # Default parameters:
    # node2vec = Node2Vec(graph, dimensions=128, walks_per_node=80, length=40, context_size=10, p=4, q=0.25)
    node2vec = Node2Vec(graph, embedding_size=10, walks_per_vertex=80,
                        walk_length=20, window=8, p=4, q=0.25)
    n2v = node2vec.train(epochs=1)
    n2v.save(output_file)

    """
    Order the embedding vectors by their index
    """
    # Create a dictionary that maps indices to keys
    key_to_index = n2v.key_to_index

    # Create a list of tuples (index, key)
    index_key_pairs = list(zip(key_to_index.keys(), key_to_index.values()))

    # Sort the list of tuples by the index
    sorted_index_key_pairs = sorted(index_key_pairs, key=lambda x: x[0])

    # Extract the word embedding vectors from the model
    embedding_vectors = n2v.vectors

    # Create a new list of the word embedding vectors, ordered by the index
    ordered_embedding_vectors = [embedding_vectors[index]
                                 for key, index in sorted_index_key_pairs]

    colors = color_communities(graph)

    """
    Visualize the result using PCA
    """
    X = ordered_embedding_vectors
    # Create a PCA object with 2 components
    pca = PCA(n_components=2)

    # Fit the PCA model to the data
    pca.fit(X)

    # Transform the data using the PCA model
    X_pca = pca.transform(X)

    ax1 = plt.subplot(1, 2, 1)
    pos = nx.spring_layout(graph, seed=14)
    nx.draw(graph, pos, with_labels=True, node_color=colors)

    ax2 = plt.subplot(1, 2, 2)
    # Create a scatter plot of the first two principal components
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)

    # Add labels to each point
    labels = np.arange(1, 35)
    for i, label in enumerate(labels):
        ax2.text(X_pca[i, 0], X_pca[i, 1], label, fontsize=8)

    # Add labels and title
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title('PCA Result')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

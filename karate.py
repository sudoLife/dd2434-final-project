from DeepWalk import DeepWalk
import scipy
import networkx as nx
import matplotlib.pyplot as plt


def main():
    matrix = scipy.io.mmread('datasets/soc-karate/soc-karate.mtx')

    # Create a graph object using networkx
    G = nx.from_scipy_sparse_array(matrix)

    window = 2
    embedding = 2
    wpw = 10
    length = 3
    epochs = 2

    deepwalk = DeepWalk(window, embedding, wpw, length, epochs)
    corpus = deepwalk.generate_corpus(G)
    model = deepwalk.train(corpus, 1)

    embedding_x = []
    embedding_y = []

    for node in G.nodes():
        result = model.wv[str(node)]
        embedding_x.append(result[0])
        embedding_y.append(result[1])

    fig, ax = plt.subplots()
    ax.scatter(embedding_x, embedding_y)

    for node in G.nodes():
        ax.annotate(node, (embedding_x[node], embedding_y[node]))

    plt.show()


if __name__ == "__main__":
    main()

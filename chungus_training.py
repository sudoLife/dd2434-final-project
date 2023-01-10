from DeepWalk import DeepWalk
from Node2Vec import Node2Vec
from NetMF import NetMF
from utils import load_data_into_graph
import numpy as np
import pickle

dataset = 'BlogCatalog'
epochs = 5

deepwalk_path = f'models/DeepWalk-{dataset}-{epochs}-epochs.npy'
node2vec_path = f'models/Node2Vec-{dataset}-{epochs}-epochs.npy'
netmf_path = f'models/NetMF-{dataset}-{epochs}-epochs.npy'


def main():
    G, _, _, _ = load_data_into_graph('BlogCatalog')

    print("Running DeepWalk:")
    deepwalk = DeepWalk(G, window=10, walk_length=40,
                        walks_per_vertex=80, embedding_size=128)

    deepwalk_corpus = deepwalk.generate_corpus()
    with open('deepwalk_corpus.pkl', 'wb') as f:
        pickle.dump(deepwalk_corpus, f)

    deepwalk_embedding = deepwalk.train(corpus=deepwalk_corpus, epochs=epochs)
    np.save(deepwalk_path, deepwalk_embedding, allow_pickle=False)

    print("Done")

    print("Running Node2Vec:")
    node2vec = Node2Vec(G, window=10, walk_length=40,
                        walks_per_vertex=80, embedding_size=128)

    node2vec_corpus = node2vec.generate_corpus()

    with open('node2vec_corpus.pkl', 'wb') as f:
        pickle.dump(node2vec_corpus, f)

    node2vec_embedding = node2vec.train(corpus=node2vec_corpus, epochs=epochs)
    np.save(node2vec_path, node2vec_embedding, allow_pickle=False)

    print("Running NetMF:")
    netmf = NetMF(G, 10, 1, 128)
    netmf_embedding = netmf.large_window_size(256)
    np.save(netmf_path, netmf_embedding, allow_pickle=False)
    print("Done")


if __name__ == "__main__":
    main()

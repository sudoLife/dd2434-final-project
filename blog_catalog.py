import networkx as nx
from DeepWalk import DeepWalk
from Node2Vec import Node2Vec
import pickle
import networkx as nx
import pandas as pd
import numpy as np

deepwalk_output_path = 'models/DeepWalk-BlogCatalog.kv'
node2vec_output_path = 'models/Node2Vec/BlogCatalog.kv'


def main():
    """
    Load the data into graph
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('datasets/BlogCatalog-dataset/data/edges.csv',
                     header=None, names=['src', 'dst'])

    # Create a graph from the DataFrame
    # 10312 nodes, 333983 edges
    G = nx.from_pandas_edgelist(
        df, source='src', target='dst', create_using=nx.Graph)

    deepwalk = DeepWalk(G, window=10, walk_length=40,
                        walks_per_vertex=80, embedding_size=128)
    node2vec = Node2Vec(G, window=10, walk_length=40,
                        walks_per_vertex=80, embedding_size=128)

    deepwalk_corpus = deepwalk.generate_corpus()
    with open('deepwalk_corpus.pkl', 'wb') as f:
        pickle.dump(deepwalk_corpus, f)
    print("Deepwalk corpus generated & dumped!")

    # NOTE: don't forget to delete this if actually training
    del deepwalk_corpus  # freeing some memory?

    node2vec_corpus = node2vec.generate_corpus()
    with open('node2vec_corpus.pkl', 'wb') as f:
        pickle.dump(node2vec_corpus, f)
    print("Node2Vec corpus generated & dumped!")
    # deepwalk_model = deepwalk.train(corpus=deepwalk_corpus, epochs=4)
    # deepwalk_model.wv.save(deepwalk_output_path)

    # node2vec_model = node2vec.train(corpus=node2vec_corpus, epochs=4)
    # node2vec_model.wv.save(node2vec_output_path)


if __name__ == "__main__":
    main()

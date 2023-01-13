import tensorflow as tf
import networkx as nx
import line1
import pandas as pd
import numpy as np
import argparse
from utils_line import DBLPDataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from color_communities import color_communities

tf.compat.v1.disable_v2_behavior()

def main():
    # Change dataset name here, run the script,
    # final embeddings will be saved to "embedding_vectors" folder
    # as .npy file
    dataset = 'Flickr'

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('datasets/' + dataset + '-dataset/data/edges.csv',
                     header=None, names=['src', 'dst'])

    # Create a graph from the DataFrame
    # 10312 nodes, 333983 edges
    G = nx.from_pandas_edgelist(
        df, source='src', target='dst', create_using=nx.Graph)

    final_embeddings = train(G, 128, isWeighted=False)    
    
    with open('Line/embedding_vectors/Line_' + dataset + '.npy', 'wb') as f:
        np.save(f, np.array(final_embeddings))


def train(graph, embedding_size, isWeighted):
    #G = DBLPDataLoader()
    #data_loader = DBLPDataLoader(graph_file=args.graph_file)
    data_loader = DBLPDataLoader(graph, isWeighted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loader', default=data_loader)
    parser.add_argument('--embedding_dim', default=embedding_size)
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='first-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=200)
    parser.add_argument('--total_graph', default=True)
    args = parser.parse_args()
    final_embedding = line1.train(args)
    #karate_line(args, final_embedding)
    return final_embedding


# Very small dataset for testing purposes
def karate_line(args, final_embedding):
    reduced_embedding = PCA(n_components=2).fit_transform(final_embedding)

    colors = color_communities(args.data_loader.g)
    G = args.data_loader.g  # the nx graph

    ax1 = plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=14)
    nx.draw(G, pos, with_labels=True, node_color=colors)

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=colors)

    for node in G.nodes():
        ax2.annotate(
            node, (reduced_embedding[node, 0], reduced_embedding[node, 1]))
    plt.show()

if __name__ == '__main__':
    main()
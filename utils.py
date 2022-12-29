import pandas as pd
import networkx as nx
import numpy as np


def load_data_into_graph(dataset_name):
    """
    This is the data loading function for BlogCatalog, Flickr, YouTube dataset
    """
    file_path = f'datasets/{dataset_name}-dataset/data'

    # Load the edges file into a pandas DataFrame
    df = pd.read_csv(f'{file_path}/edges.csv', header=None, names=['src', 'dst'])

    # Create a graph from the DataFrame
    G = nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.Graph())
    node_count = G.number_of_nodes()

    # Read the groups of each node
    group_edges = pd.read_csv(f'{file_path}/group-edges.csv', sep=',', names=['Node', 'Group'])

    # All labels
    labels = pd.read_csv(f'{file_path}/groups.csv', header=None, names=['Group'])
    label_count = len(labels)

    # Vectorize labels as array of shape (node_count , label_count)
    node_labels = np.zeros((G.number_of_nodes(), label_count), dtype=int)
    for index, row in group_edges.iterrows():
        node = row['Node']
        group = int(row['Group'])
        node_labels[node - 1][group - 1] = 1

    # plot histogram of the number of groups joined by each node
    # group_joined_count = np.sum(node_labels, axis=1)
    # plt.hist(group_joined_count)
    # plt.show()

    # Add labels as an attribute called 'groups' to each node, stored in the graph
    groups = {i + 1: node_labels[i] for i in range(node_count)}

    # Set the 'groups' attribute for all nodes
    nx.set_node_attributes(G, values=groups, name='groups')

    return G, node_count, label_count, node_labels

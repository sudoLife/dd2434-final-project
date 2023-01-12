from __future__ import print_function, division

import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx

'''
README

In default_params:
1. For directed graph, 'prop_reversed_neg': [0, 0.5, 1] 
2. For undirected graph, 'prop_reversed_neg': [0]

In the main() function:
1. Load the graph
2. Change the graph_name
3. Specify is it weighted and directed
4. Run it
'''

default_params = {
    'prop_pos': 0.5,  # Proportion of edges to remove and use as positive samples
    'prop_neg': 0.5,  # Number of non-edges to use as negative samples
    # portion of the edges in the test set that is reversed to be negative samples
    'prop_reversed_neg': [0, 0.5, 1]  # uncomment this for directed graph
    # 'prop_reversed_neg': [0]  # uncomment this for undirected graph
}


class GraphSplit():
    def __init__(self, nx_G, is_directed=False, is_weighted=False, prop_pos=default_params['prop_pos'],
                 prop_neg=default_params['prop_neg'], prop_reversed_neg=default_params['prop_reversed_neg'],
                 random_seed=42):
        self.G = nx_G
        self.G_train = nx_G  # the original graph with the positive edges in the test split removed
        self.is_directed = is_directed
        self.is_weighted = is_weighted
        self.prop_pos = prop_pos
        self.prop_neg = prop_neg
        self.prop_reversed_neg = prop_reversed_neg
        self.test_pos_edge = None
        self.test_neg_edge = None
        self._rnd = np.random.RandomState(seed=random_seed)

    def graph_preprocess(self, enforce_connectivity=True, weighted=False, directed=False):
        G = self.G

        if not weighted:
            nx.set_edge_attributes(G, 1, 'weight')

        if not directed:
            nx.to_undirected(G)

        # Take largest connected subgraph
        if enforce_connectivity and not nx.is_connected(G):
            G = max(nx.connected_component_subgraphs(G), key=len)
            print("Input graph not connected: using largest connected subgraph")

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    def get_train_test_edge_split(self):
        """
        Randomly select prop_pos of existing edges in the graph to be positive samples,
        and prop_neg of non-edges to be negative samples for the test set.
        The rest of positive and negative edges are left to be the train set.

        For directed graph, replace prop_reversed_neg of negative samples in the test set
        to be the reversion of its positive samples.

        Direction of the edges is considered.

        Remove the positive samples in test split to obtain the training graph.

        Return the generated positive and negative edges in both train and test sets.
        """

        n_edges = self.G.number_of_edges()
        n_test_pos_edges = int(self.prop_pos * n_edges)
        n_test_neg_edges = int(self.prop_neg * n_edges)

        # Find positive edges, and remove them.
        edges = list(self.G.edges())
        test_pos_edge = []
        n_count = 0
        n_ignored_count = 0
        rnd_inx = self._rnd.permutation(n_edges)
        for eii in rnd_inx:
            edge = edges[eii]

            # Remove edge from training graph
            data = self.G_train[edge[0]][edge[1]]
            self.G_train.remove_edge(*edge)

            # Check if graph is still connected
            reachable_from_v1 = nx.connected._plain_bfs(self.G_train, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G_train.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                test_pos_edge.append(edge)
                n_count += 1

            # Exit if we've found n_test_pos_edges nodes or we have gone through the whole list
            if n_count >= n_test_pos_edges:
                break

        # non_edges() already takes direction into account
        non_edges = [e for e in nx.non_edges(self.G_train)]
        print("Finding %d of %d non-edges" % (n_test_neg_edges, len(non_edges)))

        # Select m pairs of non-edges to be the negative samples for test set
        rnd_inx = self._rnd.choice(len(non_edges), n_test_neg_edges, replace=False)
        test_neg_edge = [non_edges[ii] for ii in rnd_inx]

        n_prn = len(self.prop_reversed_neg)

        self.test_pos_edge = test_pos_edge
        self.test_neg_edge = [test_neg_edge[:] for i in range(n_prn)]

        if self.is_directed:
            self.replace_neg_samples_with_reversed_pos_samples()

        test_neg_edge = self.test_neg_edge

        train_pos_edge = list(set(edges) - set(test_pos_edge))
        train_neg_edge = [None] * n_prn
        for i in range(n_prn):
            train_neg_edge[i] = list(set(non_edges) - set(test_neg_edge[i]))

        return train_pos_edge, train_neg_edge, test_pos_edge, test_neg_edge

    def replace_neg_samples_with_reversed_pos_samples(self):
        '''
        Replace a proportion of the selected non-edges to reversed positive samples
        '''

        print(f"For directed graph, start replacing the negative test samples to reversed positive samples")

        # shuffle pos_edge_list
        random.shuffle(self.test_pos_edge)

        for i, e in enumerate(self.test_pos_edge):
            # if the reversed edge is not in the pos_edge_list
            if tuple(reversed(e)) not in self.test_pos_edge:
                # replace an edge in the neg_edge_list with it
                for j, prop in enumerate(self.prop_reversed_neg):
                    if i < prop * len(self.test_pos_edge):
                        self.test_neg_edge[j][i] = tuple(reversed(e))

    def save_train_graph(self, graph_name):
        train_graph_fn = "%s%s_graph.pkl" % ("LP_data/", graph_name)
        with open(train_graph_fn, 'wb') as fp:
            pickle.dump(self.G_train, fp)
            print("Done with storing the training graph.")


def get_edge_labels(pos_edge_list, neg_edge_list):
    """
    assign label 1 to positive samples and 0 to negative samples
    return edge list and labels
    """
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return edges, labels


def main():
    # NOTE: Load the graph here, make sure to relabel so that the index start from 1 and is ordered

    # Just as an example
    file_path = f'datasets/Emails-dataset'
    df = pd.read_csv(f'{file_path}/email-Eu-core.txt', header=None, names=['src', 'dst'], sep=' ')

    G = nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.Graph())

    # NOTE: Remember to change the graph name when change dataset
    graph_name = 'Emails'

    print("Done loading the graph")

    # Get train and test edges split of the graph
    g_split = GraphSplit(G, is_directed=True, is_weighted=False)
    train_pos_edge, train_neg_edge, test_pos_edge, test_neg_edge = g_split.get_train_test_edge_split()

    # Save train_graph
    g_split.save_train_graph(graph_name)

    for i, prop in enumerate(default_params['prop_reversed_neg']):
        edges_train, labels_train = get_edge_labels(train_pos_edge, train_neg_edge[i])
        edges_test, labels_test = get_edge_labels(test_pos_edge, test_neg_edge[i])

        train_and_test_data = [edges_train, labels_train, edges_test, labels_test]
        edges_split_fn = "%s%s_edges_split_%s.pkl" % ("LP_data/", graph_name, prop)
        with open(edges_split_fn, 'wb') as fp:
            pickle.dump(train_and_test_data, fp)
            print('Done writing train and test data into file')


if __name__ == "__main__":
    main()

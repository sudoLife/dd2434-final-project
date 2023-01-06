from __future__ import print_function, division

import pickle
import argparse
import os
import numpy as np
import networkx as nx
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from DeepWalk import DeepWalk
from Node2Vec import Node2Vec

default_params = {
    'log2p': 0,  # Parameter p, p = 2**log2p
    'log2q': 0,  # Parameter q, q = 2**log2q
    'log2d': 7,  # Feature size, dimensions = 2**log2d
    'num_walks': 10,  # Number of walks from each node
    'walk_length': 80,  # Walk length
    'window_size': 10,  # Context size for word2vec
    'edge_function': "normalized_inner_product",  # Default edge function to use
    "prop_pos": 0.1,  # Proportion of edges to remove and use as positive samples
    "prop_neg": 0.1,  # Number of non-edges to use as negative samples
    #  (as a proportion of existing edges, same as prop_pos)
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
    "normalized_inner_product": lambda a, b: 1 / (1 + np.exp(-np.dot(a, b)))
    # normalized_inner_product is the one used in the paper
}


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Link Prediction")

    parser.add_argument('--input', nargs='?',
                        help='Input graph path')

    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Regenerate random positive/negative links')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--num_experiments', type=int, default=5,
                        help='Number of experiments to average. Default is 5.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


class GraphEmbedding():
    def __init__(self, nx_G=None, is_directed=False,
                 prop_pos=0.1, prop_neg=0.1, workers=1, random_seed=None):
        # TODO: nx_G is a necessary argument
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_neg
        self.prop_neg = prop_pos
        self.wvecs = None
        self.workers = workers
        self._rnd = np.random.RandomState(seed=random_seed)

    def graph_preprocess(self, enforce_connectivity=True, weighted=False, directed=False):
        G = self.G

        if not weighted:
            nx.set_edge_attributes(G, 1, 'weight')

        # TODO: Not sure if we need this, probably not
        if not directed:
            G = G.to_undirected()

        # Take largest connected subgraph
        if enforce_connectivity and not nx.is_connected(G):
            G = max(nx.connected_component_subgraphs(G), key=len)
            print("Input graph not connected: using largest connected subgraph")

        # Remove nodes with self-edges
        # I'm not sure what these imply in the dataset
        for se in nx.nodes_with_selfloops(G):
            G.remove_edge(se, se)

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))
        self.G = G

    # TODO: Add other embedding algorithms here
    def deepwalk_train_embeddings(self, window=10, walk_length=40, walks_per_vertex=80, embedding_size=128):
        deepwalk = DeepWalk(self.G, window=window, walk_length=walk_length,
                            walks_per_vertex=walks_per_vertex, embedding_size=embedding_size)

        deepwalk_model = deepwalk.train()

        self.wvecs = deepwalk_model.wv

    def node2vec_train_embeddings(self, window=10, walk_length=40, walks_per_vertex=80, embedding_size=128, p=4,
                                  q=0.25):
        node2vec = Node2Vec(self.G, window=window, walk_length=walk_length,
                            walks_per_vertex=walks_per_vertex, embedding_size=embedding_size, p=p, q=q)

        node2vec_model = node2vec.train()
        self.wvecs = node2vec_model.wv

    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.
        """
        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        n_nodes = self.G.number_of_nodes()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        # ensure the connectivity
        if not nx.is_connected(self.G):
            raise RuntimeError("Input graph is not connected")

        n_neighbors = [len(list(self.G.neighbors(v))) for v in self.G.nodes()]
        n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        non_edges = [e for e in nx.non_edges(self.G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "Only %d negative edges found" % (len(neg_edge_list))
            )

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        edges = list(self.G.edges())
        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        rnd_inx = self._rnd.permutation(n_edges)
        for eii in rnd_inx:
            edge = edges[eii]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            # Check if graph is still connected
            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                print("Found: %d    " % (n_count), end="\r")
                n_count += 1

            # Exit if we've found npos nodes or we have gone through the whole list
            if n_count >= npos:
                break

        if len(pos_edge_list) < npos:
            raise RuntimeWarning("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def edges_to_features(self, edge_list, edge_function, dimensions):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list

        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            # TODO: originally they convert v1 and v2 to strings, but I found that the key is actually int not string
            emb1 = np.asarray(self.wvecs[v1])
            emb2 = np.asarray(self.wvecs[v2])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


def create_train_test_graphs(args, graph):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.
    """
    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # Create random training and test graphs with different random edge selections
    # TODO: change the cached_fn (file name) to something automatic-generated.
    #  If not, don't forget to change it everytime you use a new graph
    # cached_fn = "%s.graph" % (os.path.basename(args.input))
    cached_fn = "karate_DeepWalk_LP.graph"
    if os.path.exists(cached_fn) and not args.regen:
        print("Loading link prediction graphs from %s" % cached_fn)
        with open(cached_fn, 'rb') as f:
            cache_data = pickle.load(f)
        Gtrain = cache_data['g_train']
        Gtest = cache_data['g_test']

    else:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = GraphEmbedding(nx_G=graph,
                                is_directed=False,
                                prop_pos=prop_pos,
                                prop_neg=prop_neg,
                                workers=args.workers)
        Gtrain.graph_preprocess(weighted=args.weighted,
                                directed=args.directed)
        Gtrain.generate_pos_neg_links()

        # Generate a different random graph for testing
        Gtest = GraphEmbedding(nx_G=graph,
                               is_directed=False,
                               prop_pos=prop_pos,
                               prop_neg=prop_neg,
                               workers=args.workers)
        Gtest.graph_preprocess(weighted=args.weighted,
                               directed=args.directed)
        Gtest.generate_pos_neg_links()

        # Cache generated  graph
        cache_data = {'g_train': Gtrain, 'g_test': Gtest}
        with open(cached_fn, 'wb') as f:
            pickle.dump(cache_data, f)

    return Gtrain, Gtest


def test_edge_functions(args, graph):
    """
    Step1: Get Gtrain and Gtest
    Step2: Get embeddings of these two graphs
    Step3: LP, calculate AUC score
    """
    Gtrain, Gtest = create_train_test_graphs(args, graph)

    p = 2.0 ** default_params['log2p']
    q = 2.0 ** default_params['log2q']
    dimensions = 2 ** default_params['log2d']
    num_walks = default_params['num_walks']
    walk_length = default_params['walk_length']
    window_size = default_params['window_size']

    # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()

    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    aucs = {name: [] for name in edge_functions}
    for iter in range(args.num_experiments):
        print("Iteration %d of %d" % (iter, args.num_experiments))

        # TODO: For now we just use DeepWalk to get the embeddings, need to write branches for different algorithms
        # Learn embeddings with current parameter values
        Gtrain.deepwalk_train_embeddings(window=window_size, walk_length=walk_length, walks_per_vertex=num_walks,
                                         embedding_size=dimensions)
        Gtest.deepwalk_train_embeddings(window=window_size, walk_length=walk_length, walks_per_vertex=num_walks,
                                        embedding_size=dimensions)

        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, dimensions)
            edge_features_test = Gtest.edges_to_features(edges_test, edge_fn, dimensions)

            # Linear classifier
            scaler = StandardScaler()
            lin_clf = LogisticRegression(C=1)
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(edge_features_train, labels_train)
            # auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

            # Test classifier
            test_predict_proba = clf.predict_proba(edge_features_test)
            auc_test = roc_auc_score(labels_test, test_predict_proba[:, 1])
            aucs[edge_fn_name].append(auc_test)

    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))

    return aucs


if __name__ == "__main__":
    args = parse_args()

    # LOAD GRAPH HERE
    G = nx.karate_club_graph()

    # # If you want to load it from the input file
    # if args.weighted:
    #     G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    # else:
    #     G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())

    test_edge_functions(args, G)

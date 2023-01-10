import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier


def kv_to_ndarray(G: nx.Graph, embedding: KeyedVectors):
    nodes = np.sort(G.nodes())
    ndarray = np.zeros(
        (len(nodes), embedding[0].shape[0]), dtype=type(embedding[0][0]))

    for node in nodes:
        ndarray[node] = embedding[str(node)]

    return ndarray


def load_data_into_graph(dataset_name):
    """
    This is the data loading function for BlogCatalog, Flickr, YouTube dataset
    """
    file_path = f'datasets/{dataset_name}-dataset/data'

    # Load the edges file into a pandas DataFrame
    df = pd.read_csv(f'{file_path}/edges.csv',
                     header=None, names=['src', 'dst'])

    # Create a graph from the DataFrame
    G = nx.from_pandas_edgelist(
        df, source='src', target='dst', create_using=nx.Graph())
    node_count = G.number_of_nodes()

    # Read the groups of each node
    group_edges = pd.read_csv(
        f'{file_path}/group-edges.csv', sep=',', names=['Node', 'Group'])

    # All labels
    labels = pd.read_csv(f'{file_path}/groups.csv',
                         header=None, names=['Group'])
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


def node_classification(G, X, y, multiple_labels=False):
    # Create the classifier
    # clf_list = {i: LogisticRegression(class_weight={0: 1, 1: 5}) for i in range(39)}
    base_clf = LogisticRegression(class_weight={0: 1, 1: 8})
    # base_clf = SVC(kernel="linear", C=0.025)
    clf = OneVsRestClassifier(base_clf)

    cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for epoch, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train, y_train)

        if multiple_labels:
            y_pred_prob = clf.predict_proba(X_test)

            # """
            # Assign weights to different group's probabilities to solve the imbalance
            # """
            # # Calculate weight of each label
            # node_count_in_each_label = np.sum(y_train, axis=0)
            # label_weight = node_count_in_each_label / np.sum(node_count_in_each_label)
            #
            # # Normalize y_pred_prob
            # y_pred_prob = y_pred_prob * label_weight / np.sum(y_pred_prob, axis=0)

            """
                Use pred_probability to select top k labels for each node
            """
            # Originally the node index starts from 1, so X[i-1] represents the embedding vector of node i
            # record the number of labels each node in the test set has
            k_labels = {i + 1: np.sum(G.nodes[i + 1]['groups'])
                        for i in test_index}

            # L is y_pred_prob[j], where test_index[j] =  real_node_index - 1 = i
            # Get the indices of the top k largest values in L
            y_pred = np.zeros((len(test_index), y.shape[1]))
            for j, i in enumerate(test_index):
                node_idx = i + 1
                k = k_labels[node_idx]
                L = y_pred_prob[j]
                sorted_arg = np.argsort(L)
                indices = sorted_arg[-k:]
                y_pred[j][indices] = 1
        else:
            y_pred = clf.predict(X_test)

        print("epoch ", epoch)
        print("f1-macro:    ", f1_score(y_test, y_pred, average='macro'))
        print("f1_micro:    ", f1_score(y_test, y_pred, average='micro'))
        print("accuracy:    ", accuracy_score(
            y_test, y_pred, normalize=True))

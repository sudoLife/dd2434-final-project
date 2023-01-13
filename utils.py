import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kv_to_ndarray(G: nx.Graph, embedding: KeyedVectors):
    nodes = np.sort(G.nodes())
    ndarray = np.zeros(
        (len(nodes), embedding[0].shape[0]), dtype=type(embedding[0][0]))

    # so that this doesn't depend on the node name damnit
    i = 0
    for node in nodes:
        ndarray[i] = embedding[str(node)]
        i += 1

    return ndarray


def load_data_into_graph(dataset_name):
    """
    This is the data loading function for BlogCatalog, Flickr, YouTube dataset
    """
    if dataset_name == 'Emails':
        df = pd.read_csv('datasets/Emails/email-Eu-core.txt',
                    header=None, names=['src', 'dst'], sep=' ')
        G = nx.from_pandas_edgelist(df, source='src', target='dst')

        # Read the groups of each node
        group_edges = pd.read_csv(f'datasets/Emails/email-Eu-core-department-labels.txt', 
                                        sep=' ', header=None)
        print(np.array(group_edges)[0])

        # All labels
        labels = np.unique(np.array(group_edges)[:,1])

        node_count = G.number_of_nodes()
        label_count = len(labels)

        #print(group_edges.shape)
        node_labels = np.array(group_edges)[:,1]
        
        # # Vectorize labels as array of shape (node_count , label_count)
        # node_labels = np.zeros((G.number_of_nodes(), label_count), dtype=int)
        # for index, row in group_edges.iterrows():
        #     #print("row shape: " + str(row[1]))
        #     node = int(row[0])
        #     group = int(row[1])
        #     node_labels[node - 1][group - 1] = 1

    else:
        file_path = f'datasets/{dataset_name}-dataset/data'

        # Load the edges file into a pandas DataFrame
        df = pd.read_csv(f'{file_path}/edges.csv',
                        header=None, names=['src', 'dst'])

        # Create a graph from the DataFrame
        G = nx.from_pandas_edgelist(
            df, source='src', target='dst', create_using=nx.Graph())

        # Read the groups of each node
        group_edges = pd.read_csv(
            f'{file_path}/group-edges.csv', sep=',', names=['Node', 'Group'])

        # All labels
        labels = pd.read_csv(f'{file_path}/groups.csv',
                            header=None, names=['Group'])

        node_count = G.number_of_nodes()
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
    groups = {i: node_labels[i] for i in range(node_count)}

    # Set the 'groups' attribute for all nodes
    nx.set_node_attributes(G, values=groups, name='groups')

    return G, node_count, label_count, node_labels


def node_classification(G, X, y, dataset, embedding_vectors_file, multiple_labels=False):
    G, node_count, label_count, node_labels = load_data_into_graph(dataset)

    # Load the embedding vectors
    input_model_file = 'Line/' + embedding_vectors_file
    X = np.load(input_model_file)
    y = node_labels  # target labels (possibly in a multi-label format)

    # Create the classifier
    base_clf = LogisticRegression(class_weight={0: 1, 1: 8})  # for single label
    clf = OneVsRestClassifier(base_clf)  # for multi label
    
    if multiple_labels:
        cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for epoch, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train, y_train)

    if multiple_labels:
        y_pred_prob = clf.predict_proba(X_test)

        # Originally the node index starts from 1, so X[i-1] represents the embedding vector of node i
        # record the number of labels each node in the test set has
        k_labels = []
        for label_vec in y_test:
            num_of_labels = np.sum(label_vec)
            k_labels.append(num_of_labels)

        # L is y_pred_prob[j], where test_index[j] =  real_node_index - 1 = i
        # Get the indices of the top k largest values in L
        y_pred = np.zeros((len(test_index), label_count))

        for i, idx in enumerate(test_index):
            L = y_pred_prob[i]  
            k = k_labels[i]
            sorted_arg = np.argsort(L)
            indices = sorted_arg[-k:]
            y_pred[i][indices] = 1

    else:  # single label
        y_pred = clf.predict(X_test)

    # Stats
    print("epoch ", epoch)
    print("f1-macro:    ", f1_score(y_test, y_pred, average='macro'))
    print("f1_micro:    ", f1_score(y_test, y_pred, average='micro'))
    print("accuracy:    ", accuracy_score(
        y_test, y_pred, normalize=True))

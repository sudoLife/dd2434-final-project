import networkx as nx
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

from max_vote import max_vote
from utils import load_data_into_graph


def main():
    # Load the data into graph
    G, node_count, label_count, node_labels = load_data_into_graph('BlogCatalog')

    """
    Split the nodes into 5-Fold for Cross-validation
    """
    # Get the nodes of the graph
    nodes = np.array(list(G.nodes()))

    macro_averaged_f1 = np.zeros(5)
    micro_averaged_f1 = np.zeros(5)

    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_index, test_index) in enumerate(kfold.split(nodes)):
        # Get the training and test data for this fold
        train_nodes, test_nodes = nodes[train_index], nodes[test_index]

        # Create a subgraph containing only the test nodes
        G_test = nx.Graph(G.subgraph(test_nodes))
        G_train = G

        # record the number of labels each node in the test set has
        k_labels = {i: np.sum(G.nodes[i]['groups']) for i in test_nodes}

        # delete the groups attribute of test set
        for node in test_nodes:
            del G_train.nodes[node]['groups']

        """
        Classification of the test nodes
        """
        prediction = {i: np.zeros(label_count) for i in test_nodes}
        for node in test_nodes:
            k = k_labels[node]
            prediction[node] = max_vote(G, node, label_count, k)

        print("classification done")

        """
        Evaluation
        """
        # Transform label of test data and prediction result to be arrays of shape (len(test_nodes),39)
        label_test = np.zeros((len(test_nodes), label_count))
        label_pred = np.zeros((len(test_nodes), label_count))
        for i, node in enumerate(test_nodes):
            label_test[i] = G_test.nodes[node]['groups']
            label_pred[i] = prediction[node]

        macro_averaged_f1[fold_idx] = metrics.f1_score(label_test, label_pred, average='macro')
        # print(f"Macro-Averaged F1 score of fold {fold_idx} : {macro_averaged_f1[fold_idx]}")

        micro_averaged_f1[fold_idx] = metrics.f1_score(label_test, label_pred, average='micro')
        # print(f"Micro-Averaged F1 score of fold {fold_idx} : {micro_averaged_f1[fold_idx]}")

    print(f"Macro-Averaged F1 score : {np.mean(macro_averaged_f1)}")
    print(f"Micro-Averaged F1 score : {np.mean(micro_averaged_f1)}")


if __name__ == "__main__":
    main()

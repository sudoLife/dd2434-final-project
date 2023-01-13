import numpy as np

from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from utils import load_data_into_graph
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

dataset = 'BlogCatalog'  # change name of dataset 
embedding_vectors = 'line_blog.npy'  # change location of your embedding vectors here

def main():
    G, node_count, label_count, node_labels = load_data_into_graph(dataset)

    # Load the embedding vectors
    input_model_file = 'Line/' + embedding_vectors
    X = np.load(input_model_file)

    #model = KeyedVectors.load(input_model_file)

    #for i, k in enumerate(model.key_to_index):
        #k = int(k)
        #X[k - 1] = model.vectors[i]
    y = node_labels  # target labels (in a multi-label format)

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
        # y_pred = clf.predict(X_test)

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
        #k_labels = {i + 1: np.sum(G.nodes[i + 1]['groups']) for i in test_index}
        k_labels = []
        for label_vec in y_test:
            num_of_labels = np.sum(label_vec)
            k_labels.append(num_of_labels)

        #for node in G.nodes(data=True):
        #    num_of_labels = np.sum(node[1]['groups'])
        #    k_labels.append(num_of_labels)

        # L is y_pred_prob[j], where test_index[j] =  real_node_index - 1 = i
        # Get the indices of the top k largest values in L
        y_pred = np.zeros((len(test_index), label_count))
        #y_pred = np.zeros((len(test_index), y.shape[1]))
        for i, idx in enumerate(test_index):
            L = y_pred_prob[i]  
            k = k_labels[i]
            sorted_arg = np.argsort(L)
            indices = sorted_arg[-k:]
            y_pred[i][indices] = 1

        """
        for j, i in enumerate(test_index):
            node_idx = i + 1
            k = k_labels[node_idx]
            L = y_pred_prob[j]
            sorted_arg = np.argsort(L)
            indices = sorted_arg[-k:]
            y_pred[j][indices] = 1
        """
        print("epoch ", epoch)
        print("f1-macro:    ", f1_score(y_test, y_pred, average='macro'))
        print("f1_micro:    ", f1_score(y_test, y_pred, average='micro'))
        print("accuracy:    ", accuracy_score(y_test, y_pred, normalize=True))

if __name__ == "__main__":
    main()

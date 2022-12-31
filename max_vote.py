import numpy as np


def max_vote(G, node, label_count, k):
    L = np.zeros(label_count)
    prediction = np.zeros(label_count)

    neighbors = G.neighbors(node)
    for i in neighbors:
        if len(G.nodes[i]) > 0:
            n_label = G.nodes[i]['groups']
            L += n_label

    # Add a random value between 0 to 1 to each of the elements in L
    # So that we can assign random label if there's less than k values that is greater than 1
    # and also choose randomly when the kth and (k+1)th most frequent labels have the same frequency
    noise = np.random.random(label_count)
    L += noise

    # Get the indices of the top k largest values in L
    sorted_arg = np.argsort(L)
    indices = sorted_arg[-k:]
    prediction[indices] = 1

    return prediction

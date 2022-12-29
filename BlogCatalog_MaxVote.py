import pandas as pd
import networkx as nx
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import metrics

from max_vote import max_vote

"""
Load the data into graph
"""
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('datasets/BlogCatalog-dataset/data/edges.csv', header=None, names=['src', 'dst'])

# Create a graph from the DataFrame
# 10312 nodes, 333983 edges
G = nx.from_pandas_edgelist(df, source='src', target='dst', create_using=nx.Graph())
node_count = G.number_of_nodes()

# Read the groups of each node
group_edges = pd.read_csv('datasets/BlogCatalog-dataset/data/group-edges.csv', sep=',', names=['Node', 'Group'])

# All labels
labels = pd.read_csv('datasets/BlogCatalog-dataset/data/groups.csv', header=None, names=['Group'])
group_count = len(labels)

# Vectorize labels as array of shape (10312 , 39)
node_labels = np.zeros((G.number_of_nodes(), group_count), dtype=int)
for index, row in group_edges.iterrows():
    node = row['Node']
    group = int(row['Group'])
    node_labels[node - 1][group - 1] = 1

# plot histogram of the number of groups joined by each node
group_joined_count = np.sum(node_labels, axis=1)
plt.hist(group_joined_count)
# plt.show()

# Add labels as an attribute called 'groups' to each node, stored in the graph
groups = {i + 1: node_labels[i] for i in range(node_count)}

# Set the 'groups' attribute for all nodes
nx.set_node_attributes(G, values=groups, name='groups')

"""
Split the BlogCatalog dataset into 80% training and 20% test set, for test set, erase all the label info
"""
# Get the nodes of the graph
nodes = list(G.nodes())

# Shuffle the nodes
random.shuffle(nodes)

# Split the nodes into a training set and a test set
train_nodes = nodes[:int(0.8 * node_count)]
test_nodes = nodes[int(0.8 * node_count):]

# Create a subgraph containing only the test nodes
G_test = nx.Graph(G.subgraph(test_nodes))

# record the number of labels each node in the test set has
k_labels = {i: np.sum(G.nodes[i]['groups']) for i in test_nodes}

# delete the groups attribute of test set
for node in test_nodes:
    del G.nodes[node]['groups']

"""
Classification of the test nodes
"""
prediction = {i: np.zeros(group_count) for i in test_nodes}
for node in test_nodes:
    k = k_labels[node]
    prediction[node] = max_vote(G, node, group_count, k)

print("classification done")

# TODO:Cross Validation

"""
Evaluation
"""
# Transform label of test data and prediction result to be arrays of shape (len(test_nodes),39)
label_test = np.zeros((len(test_nodes), group_count))
label_pred = np.zeros((len(test_nodes), group_count))
for i, node in enumerate(test_nodes):
    label_test[i] = G_test.nodes[node]['groups']
    label_pred[i] = prediction[node]

macro_averaged_f1 = metrics.f1_score(label_test, label_pred, average='macro')
print(f"Macro-Averaged F1 score : {macro_averaged_f1}")

micro_averaged_f1 = metrics.f1_score(label_test, label_pred, average='micro')
print(f"Micro-Averaged F1 score : {micro_averaged_f1}")

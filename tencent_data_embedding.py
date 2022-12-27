import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score

from Node2Vec import Node2Vec

input_file = 'datasets/tencent/train_edges.npy'
output_file = 'embedding_result/tencent.emb'

random.seed(616)
print("Loading data")
edges = np.load(input_file)
G = nx.Graph()
for i in range(169209):
    G.add_node(i)
G.add_edges_from(edges)
print("Graph created")

print("Starts training")
# Default parameters:
# node2vec = Node2Vec(graph, dimensions=128, walks_per_node=80, length=40, context_size=10, p=4, q=0.25)
node2vec = Node2Vec(G, dimensions=128, walks_per_node=5,
                    length=10, context_size=10, p=4, q=1)
n2v = node2vec.learn_features(workers=4, epochs=2)
n2v.save_word2vec_format(output_file)

# Testing calculate similarity score for Tencent dataset
pos_test = np.load('datasets/tencent/test_edges.npy')
neg_test = np.load('datasets/tencent/test_edges_false.npy')

y_true = [True] * pos_test.shape[0] + [False] * neg_test.shape[0]
X = np.vstack([pos_test, neg_test])
print('Testing...')
y_score = []
for u, v in X:
    y_score.append(n2v.wv.similarity(str(u), str(v)))

auc_test = roc_auc_score(y_true, y_score)
print('Tencent, test AUC:', auc_test)

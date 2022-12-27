import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

from gensim.models import KeyedVectors
from DeepWalk import DeepWalk

nodeNum = 169209
cpuCount = 16
savePath = 'models/tencent.kv'
load_saved = True

edges = np.load('datasets/tencent/train_edges.npy')
G = nx.Graph()
for i in range(nodeNum):
    G.add_node(i)
G.add_edges_from(edges)

if load_saved:
    wv = KeyedVectors.load(savePath)
else:
    deepwalk = DeepWalk(window=10, embedding=128,
                        walksPerVertex=10, walkLength=50, epochs=2)
    corpus = deepwalk.generate_corpus(G)
    w2v = deepwalk.train(corpus, workers=cpuCount)
    wv = w2v.wv
    wv.save(savePath)

pos_test = np.load('datasets/tencent/test_edges.npy')
neg_test = np.load(
    'datasets/tencent/test_edges_false.npy')

y_true = [True]*pos_test.shape[0] + [False]*neg_test.shape[0]
X = np.vstack([pos_test, neg_test])

print('Testing...')
y_score = []
for u, v in X:
    y_score.append(wv.similarity(str(u), str(v)))

auc_test = roc_auc_score(y_true, y_score)
print('Tencent, test AUC:', auc_test)

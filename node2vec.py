import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class SkipGramCallback(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f'Epoch #{self.epoch} start')

    def on_epoch_end(self, model):
        print(f'Epoch #{self.epoch} end')
        self.epoch += 1

    def on_train_begin(self, model):
        print("Beginning training...")

    def on_train_end(self, model):
        print("Training completed")


class Node2Vec:
    def __init__(self, graph, dimensions=128, walks_per_node=80, length=40, context_size=10, p=4, q=0.25):
        self.graph = graph
        self.dimensions = dimensions
        self.length = length
        self.walks_per_node = walks_per_node
        self.context_size = context_size
        self.p = p
        self.q = q

    def transition_prob(self, graph, p, q, v, t):
        G = graph
        probs = list()
        v_neighbors = list(G.neighbors(v))
        for x in v_neighbors:
            if t == x:
                prob = G[v][x].get('weight', 1) * (1 / p)
            elif x in G.neighbors(t):
                prob = G[v][x].get('weight', 1)
            else:
                prob = G[v][x].get('weight', 1) * (1 / q)
            probs.append(prob)
        probs = probs / np.sum(probs)
        probs = np.array(probs)
        return probs

    def node2vec_walk(self, graph, start_node, length, p, q):
        u = start_node
        G = graph
        l = length

        walk = [u]
        # generate a list of neighbors
        neighbors = list(G.neighbors(start_node))
        # we start with start_node, uniform-randomly choose one neighbour to continue (the 2nd step of the walk)
        second_node = np.random.choice(neighbors)
        walk.append(second_node)
        # after we have the first 2 steps of walk we can start the iteration

        for i in range(1, l - 1):
            t = walk[i - 1]
            v = walk[i]
            probs = self.transition_prob(G, p, q, v, t)
            neighbors = list(G.neighbors(v))
            next_node = np.random.choice(neighbors, p=probs)
            walk.append(next_node)
        return walk

    def learn_features(self, workers, epochs=2):
        # iterate through all the nodes, each generate r walks
        G = self.graph
        walks = []
        print('Random walk to get training data...')
        for i in range(self.walks_per_node):
            for start_node in G.nodes():
                if len(list(G.neighbors(start_node))) > 0:
                    # print(start_node)
                    walk = self.node2vec_walk(G, start_node, self.length, self.p, self.q)
                    walks.append(walk)
        np.random.shuffle(walks)

        callback = SkipGramCallback()
        model = Word2Vec(sentences=walks, window=self.context_size, vector_size=self.dimensions, workers=workers,
                         epochs=epochs, callbacks=[callback])
        return model.wv

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import concurrent.futures


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

    def transition_prob(self, v, t):
        probs = list()
        v_neighbors = list(self.graph.neighbors(v))
        for x in v_neighbors:
            if t == x:
                prob = self.graph[v][x].get('weight', 1) * (1 / self.p)
            elif x in self.graph.neighbors(t):
                prob = self.graph[v][x].get('weight', 1)
            else:
                prob = self.graph[v][x].get('weight', 1) * (1 / self.q)
            probs.append(prob)
        probs = probs / np.sum(probs)
        probs = np.array(probs)
        return probs

    def node2vec_walk(self, start_node):
        walk = [start_node]
        # generate a list of neighbors
        neighbors = list(self.graph.neighbors(start_node))
        # we start with start_node, uniform-randomly choose one neighbour to continue (the 2nd step of the walk)
        second_node = np.random.choice(neighbors)
        walk.append(second_node)
        # after we have the first 2 steps of walk we can start the iteration

        for i in range(1, self.length - 1):
            t = walk[i - 1]
            v = walk[i]
            probs = self.transition_prob(v, t)
            neighbors = list(self.graph.neighbors(v))
            next_node = np.random.choice(neighbors, p=probs)
            walk.append(next_node)
        return walk

    def walk_worker(self):
        local_walks = []
        for start_node in self.graph.nodes():
            if len(list(self.graph.neighbors(start_node))) > 0:
                print(start_node)
                walk = self.node2vec_walk(start_node)
                local_walks.append(walk)
        return local_walks

    def learn_features(self, workers, epochs=2):
        # iterate through all the nodes, each generate r walks
        walks = []
        print('Random walk to get training data...')
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            pool = [executor.submit(self.walk_worker)
                    for _ in range(self.walks_per_node)]

            for f in concurrent.futures.as_completed(pool):
                walks += f.result()
        # for _ in range(self.walks_per_node):
        #     for start_node in self.graph.nodes():
        #         if len(list(self.graph.neighbors(start_node))) > 0:
        #             # print(start_node)
        #             walk = self.node2vec_walk(start_node)
        #             walks.append(walk)
        np.random.shuffle(walks)

        callback = SkipGramCallback()
        model = Word2Vec(sentences=walks, window=self.context_size, vector_size=self.dimensions, workers=workers,
                         epochs=epochs, callbacks=[callback], sg=0, negative=5)
        return model.wv

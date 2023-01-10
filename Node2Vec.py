import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import concurrent.futures
from multiprocessing import cpu_count
from utils import kv_to_ndarray


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
    def __init__(self, graph, embedding_size=128, walks_per_vertex=80, walk_length=40, window=10, p=4, q=0.25, seed=42):
        self.graph = graph
        self.dimensions = embedding_size
        self.length = walk_length
        self.walks_per_node = walks_per_vertex
        self.context_size = window
        self.p = p
        self.q = q
        self.rng = np.random.default_rng(seed=seed)

    def transition_prob(self, v, t):
        v_neighbors = list(self.graph.neighbors(v))
        probs = np.zeros(len(v_neighbors))
        for i, x in enumerate(v_neighbors):
            prob = self.graph[v][x].get(
                'weight', 1)  # / (self.p if t == x else self.q)
            if t == x:
                prob *= (1 / self.p)
            elif x not in self.graph.neighbors(t):
                prob *= (1 / self.q)
            probs[i] = prob
        probs /= np.sum(probs)
        return probs

    def node2vec_walk(self, start_node):
        # generate a list of neighbors
        neighbors = list(self.graph.neighbors(start_node))
        # we start with start_node, uniform-randomly choose one neighbour to continue (the 2nd step of the walk)
        second_node = self.rng.choice(neighbors)
        walk = [str(start_node), str(second_node)]
        # after we have the first 2 steps of walk we can start the iteration

        for i in range(1, self.length - 1):
            t = int(walk[i - 1])
            v = int(walk[i])
            probs = self.transition_prob(v, t)
            neighbors = list(self.graph.neighbors(v))
            next_node = self.rng.choice(neighbors, p=probs)
            walk.append(str(next_node))
        return walk

    def walk_worker(self):
        local_walks = []
        for start_node in self.graph.nodes():
            if len(list(self.graph.neighbors(start_node))) > 0:
                walk = self.node2vec_walk(start_node)
                local_walks.append(walk)
        print("Worker finished")
        return local_walks

    def generate_corpus(self):
        corpus = []
        print('Random walk to get training data...')
        # default number of workers equals CPU count
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = [executor.submit(self.walk_worker)
                    for _ in range(self.walks_per_node)]

            for f in concurrent.futures.as_completed(pool):
                corpus += f.result()
        self.rng.shuffle(corpus)
        return corpus

    def train(self, corpus=None, workers=cpu_count(), epochs=2):
        # iterate through all the nodes, each generate r walks
        if corpus == None:
            corpus = self.generate_corpus()
        callback = SkipGramCallback()
        model = Word2Vec(corpus, window=self.context_size, vector_size=self.dimensions, workers=workers,
                         epochs=epochs, callbacks=[callback], sg=0, negative=5)

        return kv_to_ndarray(self.graph, model.wv)

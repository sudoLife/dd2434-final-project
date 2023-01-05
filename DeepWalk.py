import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import concurrent.futures
from gensim.models.callbacks import CallbackAny2Vec
from multiprocessing import cpu_count


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


class DeepWalk:
    def __init__(self, G: nx.Graph, window: int, embedding_size: int, walks_per_vertex: int, walk_length: int, seed=42) -> None:
        self.G = G
        self.window = window
        self.embedding_size = embedding_size
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        # ensuring the seed is local and the same everywhere
        self.rng = np.random.default_rng(seed)

    def generate_corpus(self) -> list:
        corpus = []
        print("Generating corpus...")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = [executor.submit(self.deep_walk)
                    for _ in range(self.walks_per_vertex)]

            for f in concurrent.futures.as_completed(pool):
                corpus += f.result()
        print("Generated.")
        return corpus

    def deep_walk(self) -> list:
        local_corpus = []
        nodes = list(self.G.nodes())
        self.rng.shuffle(nodes)

        for node in nodes:
            # generate a random walk starting at this node
            walk = self.random_walk(node)
            local_corpus.append(walk)
        print("Walk finished")
        return local_corpus

    def random_walk(self, vertex: int) -> list:
        walk = [str(vertex)]
        for _ in range(self.walk_length - 1):
            neighbors = list(self.G.neighbors(vertex))
            if len(neighbors) == 0:
                break
            weights = np.array([self.G[vertex][neighbor].get('weight', 1.0)
                                for neighbor in neighbors], dtype=np.float32)
            weights /= np.sum(weights)
            vertex = self.rng.choice(neighbors, p=weights)
            walk.append(str(vertex))
        return walk

    def train(self, corpus=None, workers=cpu_count(), epochs=2) -> Word2Vec:
        if corpus == None:
            corpus = self.generate_corpus()

        callback = SkipGramCallback()
        model = Word2Vec(
            corpus,
            vector_size=self.embedding_size,
            sg=1,  # skipgram
            hs=1,  # hierarchical softmax
            min_count=1,
            workers=workers,  # for parallel execution
            window=self.window,
            epochs=epochs,
            callbacks=[callback]
        )

        return model

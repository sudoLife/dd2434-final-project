import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import concurrent.futures
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


class DeepWalk:
    def __init__(self, G: nx.Graph, window: int, embedding: int, walks_per_vertex: int, walk_length: int, epochs: int, seed=42) -> None:
        self.G = G
        self.window = window
        self.embedding_size = embedding
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.epochs = epochs
        # ensuring the seed is local and the same everywhere
        self.rng = np.random.default_rng(seed)

    def deep_walk(self) -> list:
        local_corpus = []
        for vertex in self.G.nodes():
            # generate a random walk starting at this node
            walk = self.random_walk(vertex)
            local_corpus.append(walk)
        print("Walk finished")
        return local_corpus

    def generate_corpus(self) -> list:
        corpus = []
        print("Generating corpus...")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = [executor.submit(self.deep_walk)
                    for _ in range(self.walks_per_vertex)]

            for f in concurrent.futures.as_completed(pool):
                corpus += f.result()
        print("Generated.")
        self.rng.shuffle(corpus)
        return corpus

    def random_walk(self, vertex: int) -> list:
        walk = [str(vertex)]
        for _ in range(self.walk_length):
            neighbors = list(self.G.neighbors(vertex))
            if len(neighbors) == 0:
                break
            weights = np.array([self.G[vertex][neighbor].get('weight', 1.0)
                                for neighbor in neighbors], dtype=np.float32)
            weights /= np.sum(weights)
            vertex = self.rng.choice(neighbors, p=weights)
            walk.append(str(vertex))
        return walk

    def train(self, corpus, workers=4) -> Word2Vec:
        callback = SkipGramCallback()
        model = Word2Vec(
            corpus,
            vector_size=self.embedding_size,
            sg=1,  # skipgram
            hs=1,  # hierarchical softmax
            min_count=0,
            workers=workers,  # for parallel execution
            window=self.window,
            epochs=self.epochs,
            callbacks=[callback]
        )

        return model

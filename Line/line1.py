import tensorflow as tf
import numpy as np
import argparse
import pickle
import time
tf.compat.v1.disable_v2_behavior()
import keras
import networkx as nx

def train(args):
    graph_processor = args.graph_processor
    args.num_of_nodes = graph_processor.num_of_nodes
    model = LineModel(args)
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = graph_processor.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_optimizer, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)

                # Print training iterations in terminal, to track loss
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time = 0
                training_time = 0
            if b == (args.num_batches - 1):  # on last iteration, get final embeddings
                final_embedding = sess.run(model.embedding)
                return final_embedding  # a list of embedded nodes

class LineModel:
    def __init__(self, args):
        self.u_i = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.label = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.embedding = tf.random.uniform([args.num_of_nodes, args.embedding_dim], 
                    minval=-1, maxval=1, dtype=tf.dtypes.int32)
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = - tf.reduce_mean(tf.math.log_sigmoid(self.label * self.inner_product))
        self.learning_rate = keras.Input(dtype=tf.float32)
        
        self.optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=self.learning_rate)
        self.train_optimizer = self.optimizer.minimize(self.loss)

class GraphProcessor:
    def __init__(self, nxgraph, isWeighted):
        self.g = nxgraph #nx.karate_club_graph()
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=isWeighted)
        self.nodes_raw = self.g.nodes(data=isWeighted)
        
        self.node_index = {}
        self.node_index_reversed = {}

        if isWeighted:  # if weighted graph
            self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
            self.node_negative_distribution = np.power(
                    np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        
            for index, (node, _) in enumerate(self.nodes_raw):
                self.node_index[node] = index
                self.node_index_reversed[index] = node
            self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

        else:  # unweighted graph
            self.edge_distribution = np.ones(self.num_of_edges)
            self.node_negative_distribution = np.ones(self.num_of_nodes)

            for index, node in enumerate(self.nodes_raw):
                self.node_index[node] = index
                self.node_index_reversed[index] = node
            self.edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges_raw]

        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = NegativeSampling(prob=self.edge_distribution)

        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = NegativeSampling(prob=self.node_negative_distribution)


    def fetch_batch(self, batch_size=16, K=10):
        edge_batch_index = self.edge_sampling.sample(batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]  # undirected graph edges only, because Line-1
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    neg_node = self.node_sampling.sample()
                    if not self.g.has_edge(self.node_index_reversed[neg_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(neg_node)
                label.append(-1)
        return u_i, u_j, label

class NegativeSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]

        over = []
        under = []

        for i, U_i in enumerate(self.U):
            if U_i > 1:
                over.append(i)
            elif U_i < 1:
                under.append(i)
        while len(over) and len(under):
            i, j = over.pop(), under.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                over.append(i)
            elif self.U[i] < 1:
                under.append(i)

    def sample(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)

        r = []
        for k in range(n):
            if y[k] < self.U[i[k]]:
                r.append(i[k])
            else:
                r.append(self.K[i[k]])

        if n == 1:
            return r[0]
        else:
            return r


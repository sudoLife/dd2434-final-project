import tensorflow as tf
import numpy as np
import argparse
from utils_line import DBLPDataLoader
import pickle
import time
tf.compat.v1.disable_v2_behavior()
import keras

def train(args):
    #print(args.num_batches)
    #data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    data_loader = args.data_loader
    args.num_of_nodes = data_loader.num_of_nodes
    model = LineModel(args)
    with tf.compat.v1.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.compat.v1.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
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
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            #if b % 1000 == 0:
                #initial_embedding 
            if b == (args.num_batches - 1):  # on last iteration, get final embeddings
                final_embedding = sess.run(model.embedding)
                #normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                #print(args.num_of_nodes)
                return final_embedding  # a list of embedded nodes
                #pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            #open('data/embedding_%s.pkl' % suffix, 'wb'))

class LineModel:
    def __init__(self, args):
        # self.u_i = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        # self.u_j = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        # self.label = keras.Input(dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        # self.embedding = tf.random.uniform([args.num_of_nodes, args.embedding_dim], 
        #             minval=-1, maxval=1, dtype=tf.dtypes.int32)
        # self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        # self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        # self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        # self.loss = - tf.reduce_mean(tf.math.log_sigmoid(self.label * self.inner_product))
        # self.learning_rate = keras.Input(dtype=tf.float32)
        
        # self.optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=self.learning_rate)
        # self.train_optimizer = self.optimizer.minimize(self.loss)
            
        
        self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.embedding = tf.compat.v1.get_variable('target_embedding', [args.num_of_nodes, args.embedding_dim], 
                                    initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        elif args.proximity == 'second-order':
            self.context_embedding = tf.compat.v1.get_variable('context_embedding', [args.num_of_nodes, args.embedding_dim],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.math.log_sigmoid(self.label * self.inner_product))
        self.learning_rate = tf.compat.v1.placeholder(name='learning_rate', dtype=tf.float32)
        #self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_optimizer = self.optimizer.minimize(self.loss)
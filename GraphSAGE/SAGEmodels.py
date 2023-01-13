# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:20:42 2023

@author: Sarra
"""

import tensorflow as tf

from collections import namedtuple
from layers import Dense

from aggregators import MeanAggregator
from prediction import BipartiteEdgePredLayer

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'neigh_sampler',  # callable neigh_sampler constructor
                       'num_samples',
                       'output_dim'  # the output (i.e., hidden) dimension
                       ])


class Graphsage(object):

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, concat=True, aggregator_type="mean", identity_dim=1):

        # if identity_dim == 0 we only use the variable features as node features
        # otherwise, we use both identity features and the variable features as node features

        self.placeholders = placeholders
        self.layer_infos = layer_infos

        nb_nodes = adj.shape[0]  # number of nodes

        self.embeds = tf.compat.v1.get_variable(
            "node_embeddings", [nb_nodes, identity_dim])

        if features is None:
            if identity_dim == 0:
                raise Exception(
                    "Must have a positive value for identity feature dimension if no input features are given.")
            self.features = self.embeds  # the features are equal to node embeddings
            self.dims = [identity_dim]  # dimension of the input layer
        else:
            nb_features = features.shape[1]  # number of features
            self.features = tf.Variable(tf.constant(
                features, dtype=tf.float32), trainable=False)
            # dimension of the input layer
            self.dims = [nb_features+identity_dim]
            if identity_dim != 0:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.dims.extend(
            [layer_infos[i].num_samples for i in range(len(layer_infos))])
        self.degrees = degrees
        self.concat = concat

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        self.batch_size = placeholders["batch_size"]

        # a list of dimensions of the hidden representations from the input layer to the
        # final layer.
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.loss = 0
        self.accuracy = 0
        self.opt_op = None

        self.build()

    def build(self):
        pass

    def loss(self):
        pass

    def accuracy(self):
        pass

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        K = len(layer_infos)  # number of layers
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(K):  # for each layer
            layer = K - k - 1
            support_size *= layer_infos[layer].num_samples
            sampler = layer_infos[layer].neigh_sampler
            node = sampler((samples[k], layer_infos[layer].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
                  aggregators=None, name=None, concat=False):
        """ At each layer (or search depth k), aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer  ( or search depth k+1)
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        K = len(num_samples)  # number of layers

        # We initialize the hidden representations for all support nodes that are various hops away from target nodes
        # using original feature vectors
        # length: K + 1 where K is the number of layers
        # hidden[k] : the hidden representations of support nodes that are k hop away from target nodes
        hidden = [tf.nn.embedding_lookup(
            input_features, node_samples) for node_samples in samples]

        new_agg = aggregators is None

        if new_agg:  # if aggregators haven't been learnt yet
            aggregators = []
        for layer in range(K):  # at each layer k (search depth)
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:  # in the last layer
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x: x,
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]

            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []

            # as depth increases, the number of support nodes decreases
            for h in range(K - layer):
                # We update the hidden representations of nodes that are h hop away from target nodes
                dim_mult = 2 if concat and (layer != 0) else 1
                # if layer == 0 there is no need to concatenate since we don't have an older representation of the nodes.
                neigh_dims = [batch_size * support_sizes[h],
                              num_samples[len(num_samples) - h - 1],
                              dim_mult*dims[layer]]
                # hidden representation of nodes that are h hop away
                nodes_reps = hidden[h]
                # hidden representations of their neighbours
                neigh_reps = tf.reshape(hidden[h + 1], neigh_dims)
                next_nodes_reps = aggregator(
                    (nodes_reps, neigh_reps))  # update
                next_hidden.append(next_nodes_reps)
            hidden = next_hidden  # After updating the hidden representations of all support nodes
        return hidden[0], aggregators


class UnsupervisedGraphsage(Graphsage):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, concat=True, aggregator_type="mean", identity_dim=0):
        super(Graphsage, self).__init__(placeholders, features, adj, degrees,
                                        layer_infos)
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

    def build(self):
        labels = tf.reshape(
            tf.cast(self.placeholders['batch2'], dtype=tf.int64),
            [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        # perform "convolution"
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                                                         support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
                                          support_sizes2, aggregators=self.aggregators, concat=self.concat,
                                          model_size=self.model_size)

        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
                                                     FLAGS.neg_sample_size)
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                                             neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                                             concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                                                      dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
                                                      bilinear_weights=False,
                                                      name='edge_predict')

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

        self.loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(
            self.outputs1, self.outputs2, self.neg_outputs)
        tf.summary.scalar('loss', self.loss)

    def accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


class SupervisedGraphsage(Graphsage):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, num_classes, concat=True, aggregator_type="mean", sigmoid_loss=True,
                 identity_dim=0):

        self.inputs = placeholders["batch"]
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        super().__init__(placeholders, features, adj, degrees,
                         layer_infos, concat, aggregator_type, identity_dim)
        print(self.inputs)
        self.build()

    def build(self):
        # 1) Sample step
        samples, support_sizes = self.sample(self.inputs, self.layer_infos)

        # 2) Aggregation step
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs, self.aggregators = self.aggregate(samples, [self.features], self.dims, num_samples,
                                                        support_sizes, concat=self.concat)
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = Dense(dim_mult*self.dims[-1], self.num_classes,
                               dropout=self.placeholders['dropout'],
                               act=lambda x: x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels']))
        tf.compat.v1.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

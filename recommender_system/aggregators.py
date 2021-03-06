import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from abc import abstractmethod
import warnings
warnings.filterwarnings("ignore")

layer_id = {}


def get_layer_id(name=''):
    if name not in layer_id:
        layer_id[name] = 0
        return 0
    else:
        layer_id[name] += 1
        return layer_id[name]


class Aggregator():
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))

        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.name = name

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # neighbor_vectors[1] : (65536, 4, 32) ===> (65536, 1, 4, 32)
        # neighbor_vectors[2] : (65536, 16, 32) ===> (65536, 4, 4, 32)

        # relation_vectors[0] : (65536, 4, 32) ===> (65536, 1, 4, 32)
        # relation_vectors[1] : (65536, 16, 32) ===> (65536, 4, 4, 32)

        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            # user_embeddings: [65536, 32]==>[65536,1,1,32]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, 1, 1, dim]
            # [65536, 1, 1, 32] * [65536, 1, 4, 32] ===> [65536, 1, 4]
            # [65536, 1, 1, 32] * [65536, 4, 4, 32] ===> [65536, 4, 4]
            # ??????R?????????u??????????????????
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)

            # ???????????????
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)
            # ??????
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)
            # ??????????????????????????????????????????????????????????????????????????????????????????????????????
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0, act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.truncated_normal_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.truncated_normal_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.truncated_normal_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)






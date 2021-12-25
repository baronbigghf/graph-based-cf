import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, entity_matrix, relation_matrix):
        self._parse_args(args, entity_matrix, relation_matrix)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.truncated_normal_initializer()

    def _parse_args(self, args, entity_matrix, relation_matrix):
        self.entity_matrix = entity_matrix
        self.relation_matrix = relation_matrix

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = 4
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_index = tf.placeholder(dtype=tf.int64, shape=[None], name='user_index')
        self.item_index = tf.placeholder(dtype=tf.int64, shape=[None], name='item_index')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_embedding_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_embedding_matrix'
        )
        self.entity_embedding_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_embedding_matrix'
        )
        self.relation_embedding_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_embedding_matrix'
        )

        self.user_embeddings = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_index)
        entities, relations = self.get_neighbors(self.item_index)

        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.entity_matrix, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.relation_matrix, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_embedding_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_embedding_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_embedding_matrix) + \
                       tf.nn.l2_loss(self.entity_embedding_matrix) + \
                       tf.nn.l2_loss(self.relation_embedding_matrix)

        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, fedd_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], fedd_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores < 0.5] = 0
        scores[scores >= 0.5] = 1
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_index, self.scores_normalized], feed_dict)







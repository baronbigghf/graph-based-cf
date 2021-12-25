import os
import numpy as np



def load_data():
    """

    :param:
    :return:
        - n_user: user数 13159
        - n_item: item数 16954
        - n_entity： 知识图谱中实体数量 102568
        - n_relation： 知识图谱中关系数量 32
        - train_data, eval_data, test_data： ratings_final.txt中的测试集合
        - entity_matrix， relation_matrix，
    """
    n_user, n_item, train_data, eval_data, test_data = load_rating()
    n_entity, n_relation, entity_matrix, relation_matrix = load_kg()
    print("data loading ...")

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, entity_matrix, relation_matrix


def load_rating():
    print("reading ratings ... ")

    # reading rate
    path = '/Users/zenki/PycharmProjects/recommender_system/dataset'
    rating_file = path + '/ratings_final'

    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    print("n_user:", n_user)
    n_item = len(set(rating_np[:, 1]))
    print("n_item:", n_item)

    train_data, eval_data, test_data = dataset_slicer(rating_np)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_slicer(rating_np):
    print("dividing training set & test set ...")

    # train:eval:test = 7:2:1
    eval_ratio = 0.2
    test_ratio = 0.1
    n_ratings = rating_np.shape[0]
    print("n_ratings=", n_ratings)

    # random
    eval_index = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False, p=None)
    left = set(range(n_ratings)) - set(eval_index)
    test_index = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False, p=None)
    train_index = list(left - set(test_index))

    train_data = rating_np[train_index]
    eval_data = rating_np[eval_index]
    test_data = rating_np[test_index]

    return train_data, eval_data, test_data


def load_kg():
    print("reading knowledge graph ...")

    # reading kg
    path = '/Users/zenki/PycharmProjects/recommender_system/dataset'
    kg_file = path + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    # 头实体head & 尾实体 tail
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    entity_matrix, relation_matrix = construct_matrix(kg, n_entity)

    return n_entity, n_relation, entity_matrix, relation_matrix


def construct_kg(kg_np):
    print("constructing knowledge graph ...")

    kg = dict()
    for triple in kg_np:
        head_entity = triple[0]
        relation = triple[1]
        tail_entity = triple[2]

        if head_entity not in kg:
            kg[head_entity] = []
        kg[head_entity].append((tail_entity, relation))
        if tail_entity not in kg:
            kg[tail_entity] = []
        kg[tail_entity].append((head_entity, relation))

    return kg


def construct_matrix(kg, entity_num):
    print("constructing entity matrix ...")

    neighbor_size = 4
    entity_matrix = np.zeros([entity_num, 4], dtype=np.int64)
    relation_matrix = np.zeros([entity_num, 4], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= 4:
            sampled_index = np.random.choice(list(range(n_neighbors)), size=4, replace=False, p=None)
        else:
            sampled_index = np.random.choice(list(range(n_neighbors)), size=4, replace=True, p=None)
        entity_matrix[entity] = np.array([neighbors[i][0] for i in sampled_index])
        relation_matrix[entity] = np.array([neighbors[i][1] for i in sampled_index])

    return entity_matrix, relation_matrix










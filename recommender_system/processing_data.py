import numpy as np


def read_item_index2entity():
    path = '/Users/zenki/PycharmProjects/recommender_system/dataset'
    file = path + '/item_index2entity_id.txt'
    print("reading item_index2entity_id.txt ...")
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i = i + 1


def convert_rating():
    path = '/Users/zenki/PycharmProjects/recommender_system/dataset'
    file = path + '/ratings.csv'
    print("reading rating.csv ...")
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(',')

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= 4:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print("converting ratings_final.txt ...")
    writer = open(path + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new [user_index_old] = user_cnt
            user_cnt = user_cnt + 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t1\n' % (user_index, item))
    writer.close()
    print('nums of users:', user_cnt)
    print('nums of items', len(item_set))


def convert_kg():
    print('converting kg file ...')
    path = '/Users/zenki/PycharmProjects/recommender_system/dataset'
    file = path + '/kg_final.txt'
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open(file, 'w', encoding='utf-8')
    for line in open(path + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt = entity_cnt + 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt = entity_cnt +1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt = relation_cnt + 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('nums of entities (containing items)',  entity_cnt)
    print('nums of relations:', relation_cnt)


if __name__ == '__main__':
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index2entity()
    convert_rating()
    convert_kg()

    print('done')

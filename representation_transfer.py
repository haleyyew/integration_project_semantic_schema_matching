
def transfer_mapping_to_topic_per_table(many_to_many_mappings, tables):
    final_output = {}
    for table in tables:
        topics_output = []
        table_mappings = get_pairs_for_table(many_to_many_mappings, table)

        for topic in table_mappings:
            if len(table_mappings[topic]) != 0:
                topics_output.append(topic)

        final_output[table] = topics_output

    return final_output

def get_pairs_for_table(many_to_many_mappings, table_name):
    table_mappings = {}
    for topic in many_to_many_mappings:
        values = []
        for value in many_to_many_mappings[topic]:
            if value[0] == table_name:
                values.append(value)
        table_mappings[topic] = values

    return table_mappings

import itertools
import pprint
def enumerate_one_to_one_correspondences(many_to_many_mappings, tables):
    set_of_one_to_one = {}

    for table in tables:
        table_mappings = get_pairs_for_table(many_to_many_mappings, table)
        input = []
        topics_input = []
        for topic in table_mappings:
            input.append(table_mappings[topic]+[(table, None, 0)])
            topics_input.append(topic)
        dupl = list(itertools.product(*input))
        unique = []
        for matching in dupl:
            duplicates = {}
            for item in matching:
                if item[1] not in duplicates:
                    duplicates[item[1]] = 1
                else:
                    duplicates[item[1]] += 1

            add = True
            for item in duplicates:
                if item != None and duplicates[item] > 1:
                    add = False

            if add:
                unique.append(matching)

        # pprint.pprint(unique)
        set_of_one_to_one[table] = (unique, topics_input)

    return set_of_one_to_one

def tuple_to_mapping(set_of_table_tuple, tables, topics):
    mappings = {}
    for topic in topics:
        mappings[topic] = []

    for table in tables:
        table_mappings = set_of_table_tuple[table]
        for i in range(len(table_mappings)):
            if table_mappings[i][1] == None:
                continue
            mappings[topics[i]].append(table_mappings[i])
    return mappings

many_to_many_mappings = {'A':[('x','b',0.8),('x','c',0.6),('y','e',0.5)], 'B':[('x','b',0.6),('x','c',0.8),('y','e',0.6)], 'C':[('x','c',0.8),('x','d',0.5),('y','f',0.5)]}
topics = ['A','B','C']
tables = ['x','y']
table1_attr = ['b','c','d']
table2_attr = ['e','b']


set_of_one_to_one = enumerate_one_to_one_correspondences(many_to_many_mappings, tables)
pprint.pprint(set_of_one_to_one)


set_of_table_tuple = {'x': (('x', 'b', 0.8), ('x', 'c', 0.8), ('x', 'd', 0.5)), 'y': (('y', None, 0), ('y', None, 0), ('y', 'f', 0.5))}

# test_mappings = {'A':[('x','b',0.8),('x','c',0.6),('y','e',0.5)], 'B':[('x','b',0.6),('x','c',0.8)]}

test_mappings = tuple_to_mapping(set_of_table_tuple, tables, topics)

final_output = transfer_mapping_to_topic_per_table(test_mappings, tables)
pprint.pprint(final_output)
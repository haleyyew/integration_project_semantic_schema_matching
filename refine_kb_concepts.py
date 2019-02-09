DEBUG_MODE = False

import json
import os
import csv
import pandas as pd

def add_attr_to_kb_concept(kb, matching_context):
    concept = matching_context['concept']
    concept_dict = kb[matching_context['concept']]['matches']
    src_attr = matching_context['src_attr']
    src_dataset = matching_context['src_dataset']
    tar_dataset = matching_context['tar_dataset']
    tar_attr = matching_context['tar_attr']
    match_score = matching_context['score'],
    example_values = matching_context['example_values'],
    data_type = matching_context['data_type']

    print('=add_attr_to_kb_concept= concept:%s src:%s(%s)<=%0.2f=>%s(%s)' % (concept, src_dataset, src_attr, float(match_score[0]), tar_dataset, tar_attr))

    if 'cluster' not in concept_dict[src_dataset]:
        concept_dict[src_dataset]['cluster'] = {}
    cluster = concept_dict[src_dataset]['cluster']
    # TODO update existing if necessary
    if concept_dict[src_dataset]['attribute'] != src_attr:
        print('change kb: src_attr %s->%s' % (concept_dict[src_dataset]['attribute'], src_attr))
        concept_dict[src_dataset]['attribute'] = src_attr
    cluster[tar_dataset] = {
        'attribute': tar_attr,
        'match_score': float(match_score[0]),
        'example_values': [item for sublist in list(example_values) for item in sublist] ,
        'data_type': data_type
    }

    datasources = kb[matching_context['concept']]['datasources']
    datasources.append(tar_dataset)
    kb[matching_context['concept']]['datasources'] = list(set(datasources))

    return

def get_example_values_from_schema(schema_set, datasource, attr_key):
    attrs_schema = []
    for attr in schema_set[datasource]:
        name = attr['name']
        attrs_schema.append(name)

    idx = attrs_schema.index(attr_key)
    attr_vals = schema_set[datasource][idx]['coded_values']
    attr_datatype = schema_set[datasource][idx]['data_type']

    return attr_vals, attr_datatype

import nltk
nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
from nltk.corpus import wordnet
import inflection
def similarity_between_wordpair(src_word, tar_word):
    sem_sim_score = 0
    src_word_vec = src_word.split(' ')
    tar_word_vec = tar_word.split(' ')
    for word1 in tar_word_vec:
        word1 = inflection.singularize(word1)

        w1 = None
        try:
            w1 = wordnet.synset(word1+'.n.01')
        except Exception:
            continue

        for word2 in src_word_vec:
            word2 = inflection.singularize(word2)

            w2 = None
            try:
                w2 = wordnet.synset(word2 + '.n.01')
            except Exception:
                continue

            sem_sim = w1.wup_similarity(w2)
            sem_sim_score += sem_sim
            # print(word1, word2, sem_sim)

    sem_sim_score = sem_sim_score/(len(src_word_vec) * len(tar_word_vec))
    return sem_sim_score

def find_similarity_between_wordsets(concept, src_word, tar_words):
    sem_sims = {}
    sem_sims[(concept, src_word)] = similarity_between_wordpair(concept, src_word)

    src_tar_sims = 0
    src_tar_pairs = 0
    for tar_word in tar_words:
        sem_sims[(src_word, tar_word)] = similarity_between_wordpair(src_word, tar_word)
        src_tar_pairs += 1
        src_tar_sims += sem_sims[(src_word, tar_word)]

    src_tar_score = src_tar_sims/src_tar_pairs

    if src_tar_score >= sem_sims[(concept, src_word)]:
        # stay in concept
        return False, sem_sims, src_tar_score
    else:
        # leave concept, create new concept
        return True, sem_sims, src_tar_score


import pprint
import numpy as np
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
import scipy


def find_all_subtree_mappings(root, testing):
    '''
    For comparing clusters:
    Extract mappings first, represent in vectors, then perform clustering for mappings
    Split topic into two if two distinct clusters are found within topic, can split into more
    Merge clusters into one topic if two clusters from two topics have similar mappings
    '''
    mappings = []
    # mapping_pairs = {}

    for concept in root:
        matches = root[concept]['matches']
        if testing:
            matches = root[concept]
        for dataset in matches:
            if 'cluster' not in matches[dataset]:
                continue
            for mapped_dataset in matches[dataset]['cluster']:
                mapping = [concept,
                             dataset,
                             mapped_dataset,
                             matches[dataset]['attribute'],
                             matches[dataset]['cluster'][mapped_dataset]['attribute'],
                             matches[dataset]['cluster'][mapped_dataset]['match_score']]
                mappings.append(mapping)

                fwd_attrs = dataset+'AND'+mapped_dataset
                bkwd_attrs = mapped_dataset+'AND'+dataset
                # if fwd_attrs in mapping_pairs:
                #     mapping_pairs[fwd_attrs] += 1
                # elif bkwd_attrs in mapping_pairs:
                #     mapping_pairs[bkwd_attrs] += 1
                # else:
                #     mapping_pairs[fwd_attrs] = 0

    # pprint.pprint(mappings)

    return mappings

def hierarchical_cluster_linkage(features, decision_threshold):
    if DEBUG_MODE: print('hierarchical_cluster_linkage:')
    if DEBUG_MODE: print(features)
    if len(features) < 2:
        return []

    arr = scipy.array(features)
    pd = ssd.pdist(arr, metric='cosine')
    z = hac.linkage(pd)

    if DEBUG_MODE: pprint.pprint(pd)

    part = hac.fcluster(z, decision_threshold, 'distance')
    return part

def hierarchical_cluster(scores, a_keys, decision_threshold):
    if DEBUG_MODE: print('hierarchical_cluster dist:', scores)

    if len(scores) < 2:
        return []

    a_len = len(a_keys)
    # a = [a[key] for key in a_keys]
    a = np.zeros(shape=(a_len, a_len))

    k = 0
    for i in range(a_len):
        for j in range(i + 1, a_len):
            a[i, j] = scores[k]
            a[j, i] = scores[k]
            # print(a[i, j], a[j, i])
            k += 1

    # a = np.array([[0, 0, 2, 2],
    #               [0, 0, 2, 2],
    #               [2, 2, 0, 0],
    #               [2, 2, 0, 0]])

    a = ssd.squareform(a)
    if DEBUG_MODE: print(a)

    z = hac.linkage(a)

    part = hac.fcluster(z, decision_threshold, 'inconsistent')
    # print(part)

    # for cluster in set(part):
    #     print(cluster)

    return part


# need to know all concepts in root so far
# do N! comparisons of mappings
def split_concepts(root, mappings, decision_threshold):
    '''split to a new temp concept'''

    mappings = pd.DataFrame(mappings, columns=['concept', 'src_dataset', 'tar_dataset', 'src_attr', 'tar_attr', 'score'])
    concepts = list(root.keys())

    # print(concepts)

    for concept in concepts:
        # filter mappings for the concept
        # concept_mappings = [tuple for tuple in mappings if tuple[0] == concept]
        concept_mappings = mappings.loc[mappings['concept'] == concept]

        if DEBUG_MODE: print(concept_mappings)

        clusters = concept_mappings.groupby(['src_dataset', 'src_attr'])
        keys = list(clusters.groups.keys())
        num_keys = len(keys)

        clusters_to_split = {}
        clusters_to_split_list = []
        for i in range(num_keys):
            key_i = keys[i]
            cluster_i = clusters.get_group(key_i)

            # print(key_i, cluster_i)

            list_i = []
            for index, row in cluster_i.iterrows():
                val = row['tar_dataset'] + '.' + row['tar_attr']
                list_i.append(val)
            num_i = len(list_i)

            for j in range(i+1, num_keys):
                key_j = keys[j]
                cluster_j = clusters.get_group(key_j)

                list_j = []
                for index, row in cluster_j.iterrows():
                    val = row['tar_dataset'] + '.' + row['tar_attr']
                    list_j.append(val)
                num_j = len(list_j)

                diff = set(list_i).symmetric_difference(set(list_j))

                diff_score = len(list(diff))/(num_i+num_j)
                # print(diff)
                diff_score = diff_score if diff_score > decision_threshold else 0
                if (key_i, key_j) in clusters_to_split:
                #     clusters_to_split[(key_i, key_j)] += diff_score
                # elif (key_j, key_i) in clusters_to_split:
                #     clusters_to_split[(key_j, key_i)] += diff_score
                    pass
                else:
                    clusters_to_split[(key_i, key_j)] = diff_score
                    clusters_to_split_list.append(diff_score)
                    # print(key_i, key_j)

        # pprint.pprint(clusters_to_split)
        part = hierarchical_cluster(clusters_to_split_list, keys, decision_threshold)
        if DEBUG_MODE: print('split_concepts: ', 'part', part, 'keys', keys)

        # find clusters to split, then split to new concept
        if len(part) > 1 and len(list(set(part))) > 1:
            num_new_concepts = len(list(set(part))) - 1
            new_concepts = {'temp_'+concept+'_'+str(i+2) : {} for i in range(num_new_concepts)}
            print('new concepts:', new_concepts)

            for j in range(len(keys)):
                key = keys[j]
                partition = part[j]
                if partition != 1:
                    print('temp_'+concept+'_'+str(partition))
                    new_concepts['temp_'+concept+'_'+str(partition)][key[0]] = root[concept][key[0]]
                    del root[concept][key[0]]

            root.update(new_concepts)

    return root




def merge_concepts(root, mappings, decision_threshold):
    '''merge concepts; rename temp concepts not merged this iteration'''
    # merge clusters from different concepts, (or move cluster from first concept to second concept)
    # must also add parent attr to keys, form cluster of attributes
    # when instance matching, compare all values in merged cluster
    # TODO update concept-attr match score, update temp_concept names

    # pprint.pprint(mappings)
    if DEBUG_MODE: print('merge_concepts:')

    mappings = pd.DataFrame(mappings, columns=['concept', 'src_dataset', 'tar_dataset', 'src_attr', 'tar_attr', 'score'])
    concepts = list(root.keys())

    clusters = mappings.groupby(['concept', 'src_dataset', 'src_attr'])
    keys = list(clusters.groups.keys())
    num_keys = len(keys)
    # print(num_keys)

    # all clusters must share one edge with each other
    # number of attrs shared must be at least decision_threshold to be considered for merging
    # decide which cluster to merge into

    # key is sorted tuple of datasource.attr
    clusters_to_merge = {}
    features = []
    features_row_keys = []

    # collect all attrs in mappings as cols in matrix
    all_attrs = []
    for i in range(num_keys):
        key_i = keys[i]
        cluster_i = clusters.get_group(key_i)
        all_attrs.append(key_i[1] + 'DOT' + key_i[2])

        for index, row in cluster_i.iterrows():
            val = row['tar_dataset'] + 'DOT' + row['tar_attr']
            all_attrs.append(val)

    # print(all_attrs)
    all_attrs = list(set(all_attrs))
    all_attrs.sort()
    all_attrs_keys = {}
    all_attrs_len = len(all_attrs)
    for i in range(all_attrs_len):
        all_attrs_keys[all_attrs[i]] = i
    # print(all_attrs_keys)
    len_keys = len(all_attrs_keys.keys())
    # print(len_keys)

    for i in range(num_keys):
        key_i = keys[i]
        cluster_i = clusters.get_group(key_i)

        feature_vec = [0] * len_keys

        # attr with concept also included
        index = all_attrs_keys[key_i[1] + 'DOT' + key_i[2]]
        feature_vec[index] = 1

        for index, row in cluster_i.iterrows():
            index = all_attrs_keys[row['tar_dataset'] + 'DOT' + row['tar_attr']]
            feature_vec[index] = 1

        features.append(feature_vec)
        features_row_keys.append((key_i[0], key_i[1], key_i[2]))

    if DEBUG_MODE: pprint.pprint(features)

    part = hierarchical_cluster_linkage(features, decision_threshold)
    if DEBUG_MODE: pprint.pprint(part)

    # features_df = pd.DataFrame(features, index=features_row_keys, columns=all_attrs)
    # print(features_df.to_string())

    if len(part) <= 1 or len(list(set(part))) <= 1:
        return root

    part_indexes = [[part[i], i] for i in range(len(part))]
    part_indexes_df = pd.DataFrame(part_indexes, columns=['group', 'index'])
    # print(part_indexes_df.to_string())

    groups_df = part_indexes_df.groupby('group')['index'].apply(list)
    # print(groups_df.to_string())

    for column in groups_df:
        # print(column)
        if len(column) > 1:
            print("merging:", column)
            concept_name = ''
            temp_concept_subtrees = {'temp':{}}
            for i in column:
                concept = features_row_keys[i][0]
                if 'temp' not in concept:
                    concept_name = concept_name + '+' + concept

                temp_concept_subtrees['temp'][features_row_keys[i][1]] = root[concept][features_row_keys[i][1]]
                del root[concept][features_row_keys[i][1]]

            root[concept_name] = temp_concept_subtrees['temp']


    return root

def create_new_kb_concept(attr_schema_parse, root, testing):
    new_concepts = {}

    for ds in list(attr_schema_parse.keys()):
        mappings_datasource = traverse_tree_for_attrs(root, ds, testing)
        # print(ds)
        # pprint.pprint(mappings_datasource)

        attrs = []
        for mapping in mappings_datasource:
            attrs.append(mapping[1])

        attrs = list(set(attrs))
        # print(attrs)

        # find attrs not in map
        diff = set(attr_schema_parse[ds]).symmetric_difference(set(attrs))
        diff = list(diff)
        # print(diff)

        for new_attr in diff:
            if new_attr in new_concepts:
                new_concepts[new_attr].append(ds)
            else:
                new_concepts[new_attr] = [ds]

    return new_concepts

def traverse_tree_for_attrs(root, datasource, testing):
    '''For finding new concepts'''
    mappings = []

    for concept in root:
        # print(concept)
        matches = root[concept]['matches']
        if testing:
            matches = root[concept]

        for dataset in matches:

            # print(dataset)
            dataset_attr = matches[dataset]['attribute']
            dataset_attr_score = matches[dataset]['match_score']
            if dataset == datasource:
                mappings.append((concept, dataset_attr, dataset_attr_score))

            if 'cluster' not in matches[dataset]:
                continue
            for mapped_dataset in matches[dataset]['cluster']:
                mapped_dataset_attr = matches[dataset]['cluster'][mapped_dataset]['attribute']
                mapped_dataset_attr_score = matches[dataset]['cluster'][mapped_dataset]['match_score']

                if mapped_dataset == datasource:
                    mappings.append((concept, mapped_dataset_attr, mapped_dataset_attr_score))
    return mappings

def find_cluster_semantic_relatedness(kb):

    for concept in kb:
        matches = kb[concept]['matches']
        for src_dataset in matches:
            if 'cluster' not in matches[src_dataset]:
                continue
            cluster = matches[src_dataset]['cluster']
            src_attr = matches[src_dataset]['attribute']
            cluster_attrs = []
            for tar_dataset in cluster:
                attr = cluster[tar_dataset]['attribute']
                cluster_attrs.append(attr)
            leave_concept, sem_sims, _ = find_similarity_between_wordsets(concept, src_attr, cluster_attrs)
            print(leave_concept, sem_sims)

            # if leave_concept:
            #     create_new_kb_concept(kb)
    # TODO incorporate into module for examining mapping clusters

    return

def select_new_concepts(metadata_set, schema_set, input_new_concepts, kb, num):
    output_new_concepts = []

    # policy 1: select topics related to current topics in kb, chance of introducing new datasets
    existing_concepts = list(kb.keys())
    all_concepts = list(metadata_set['tags'].keys())
    all_concepts = list(set(all_concepts) - set(existing_concepts))

    concept_sims = {}
    columns = ['concept', 'score']
    columns.extend(existing_concepts)
    concept_sims_scores = pd.DataFrame(columns=columns)
    concept_sims_scores_series = []
    for concept in all_concepts:
        # print(concept)
        _, sem_sims, score = find_similarity_between_wordsets(concept, concept, existing_concepts)
        # print(score)
        concept_sims[concept] = sem_sims
        series_data = [concept, score]
        series_data.extend([sem_sims[(concept, kb_conc)] for kb_conc in existing_concepts])
        # print(series_data)
        # print(pd.Series(series_data, index=concept_sims_scores.columns))
        concept_sims_scores = concept_sims_scores.append(pd.Series(series_data, index=concept_sims_scores.columns), ignore_index=True)
        # print(concept_sims_scores.head(2))
        # print('-----')

    if DEBUG_MODE: print(concept_sims_scores.head(20))
    # print('-----')

    top_concepts = concept_sims_scores.sort_values('score', ascending=False).head(num)
    # print(top_concepts)
    output_new_concepts = list(top_concepts['concept'])

    # TODO policy 2: select most popular attributes from current mapped datasets, chance of creating new topics

    # TODO policy 3: select attributes not from current mapped datasets but related to topics in kb, chance of discovering new information

    return input_new_concepts, output_new_concepts, concept_sims_scores, concept_sims_scores

if __name__ == "__main__":
    kb_f = open('./kb_file.json', 'r')
    kb = json.load(kb_f)
    instance_matching_outputs_path = './instance_matching_output'
    match_score_threshold = 10

    schema_f = open('./schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)

    for root, dirs, files in os.walk(instance_matching_outputs_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.csv':
                continue
            output_df = pd.read_csv(root + '/' + file, index_col=0, header=0)
            tokens = filename.split('|')
            concept = tokens[0]
            src_dataset = tokens[1]
            src_attr = tokens[2]
            tar_dataset = tokens[4]

            print("[concept:%s, src:%s(%s) <=> tar:%s]" % (concept, src_dataset, src_attr, tar_dataset))
            # print(output.head())
            columns = list(output_df.columns.values)
            if len(columns) == 0:
                continue

            indexes = list(output_df.index.values)
            best_matches = {src_attr : (None, 0) for src_attr in indexes}
            for attr in columns:
                for index, row in output_df.iterrows():
                    score = row[attr]
                    src_attr = index
                    tar_attr = attr
                    if score > best_matches[src_attr][1]:
                        best_matches[src_attr] = (tar_attr, score)

            for src_attr in best_matches:
                score = float(best_matches[src_attr][1])
                tar_attr = best_matches[src_attr][0]
                print('best matches: %s <=%0.2f=> %s' % (src_attr, score, tar_attr))

                if score > match_score_threshold:
                    attr_vals, attr_datatype = get_example_values_from_schema(schema_set, tar_dataset, tar_attr)
                    matching_context = {
                        'concept': concept,
                        'src_attr': src_attr,
                        'src_dataset': src_dataset,
                        'tar_dataset' : tar_dataset,
                        'tar_attr': tar_attr,
                        'score': score,
                        'example_values': attr_vals,
                        'data_type': attr_datatype
                    }
                    add_attr_to_kb_concept(kb, matching_context)



    # TODO look at all datasources, pick some attributes not covered in kb as new concepts

    kb_file = open("kb_file.json", "w")
    json.dump(kb, kb_file, indent=2, sort_keys=True)


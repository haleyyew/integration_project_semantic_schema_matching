import os
import sys
import networkx
import itertools
import pandas as pd
from numpy import array

scriptpath = "./parse_dataset.py"
sys.path.append(os.path.abspath(scriptpath))
import parse_dataset

import math
import collections

import sklearn.cluster as cluster
import numpy as np
import matplotlib.pyplot

import pprint
pp = pprint.PrettyPrinter(indent=2)


class GraphicalModel(object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.texts_vectors = []
        self.edge_ids = []
        self.similarities = []


    def cluster(self):
        similarities_array = np.asarray(self.similarities, dtype=np.float32)

        k_means = cluster.KMeans(n_clusters=2)
        k_means.fit(similarities_array.reshape(-1, 1))
        # print(k_means.labels_)

        for j in range(len(k_means.labels_)):
            label = k_means.labels_[j]
            similarity = self.similarities[j]
            edge = self.edges[self.edge_ids[j]]
            print('(', edge.node0, edge.node1, ')', similarity, ':', label)


def build_graph(data_model):
    graphical_model = GraphicalModel()

    nodes = {}
    edges = {}

    texts_vectors = []

    for key in data_model.datasets:
        data_instance = data_model.datasets[key]
        print('Data Instance: ' + key)

        for resource in data_instance.resources:
            print(resource['format'])

            data = resource['data']
            first_row = data[0]
            for attribute_name in first_row:
                # print('\t' + attribute_name)

                text = attribute_name
                textvec = word2vec(attribute_name)

                node = AttributeNode()
                node.resource_name = resource
                node.attribute_name_vec = textvec
                node.attribute_name = text

                texts_vectors.append(node.id)
                nodes[node.id] = node

    S = complete_graph_from_list(texts_vectors)

    edge_ids = []
    similarities = []
    for key in S.edges.keys():
        # print('key', key)
        # print(key[0], 'and', key[1])
        # print('value', S.edges.get(key))
        edge = AttributeEdge()
        edge.node0 = key[0]
        edge.node1 = key[1]
        # print(nodes[edge.node0].attribute_name)
        # print(nodes[edge.node1].attribute_name)
        edge.similarity = cosdis(nodes[edge.node0].attribute_name_vec, nodes[edge.node1].attribute_name_vec)
        edges[edge.id] = edge
        # print('(', edge.node0, edge.node1, ')',  edges[key].similarity)
        edge_ids.append(edge.id)
        similarities.append(edge.similarity)

    graphical_model.nodes = nodes
    graphical_model.edges = edges
    graphical_model.texts_vectors = texts_vectors
    graphical_model.edge_ids = edge_ids
    graphical_model.similarities = similarities

    return graphical_model

def complete_graph_from_list(L, create_using=None):
    G = networkx.empty_graph(len(L),create_using)
    if len(L)>1:
        if G.is_directed():
            edges = itertools.permutations(L,2)
        else:
            edges = itertools.combinations(L,2)
        G.add_edges_from(edges)
    return G


def word2vec(word):
    # count the characters in word
    cw = collections.Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = math.sqrt(sum(c*c for c in cw.values()))

    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


class AttributeNode(object):
    newid = 0

    def __init__(self):
        self.id = AttributeNode.newid
        AttributeNode.newid += 1
        self.resource_name = ''
        self.attribute_name = ''
        self.attribute_name_vec = ''

class AttributeEdge(object):
    newid = 0

    def __init__(self):
        self.id = AttributeEdge.newid
        AttributeEdge.newid += 1

        self.node0 = -1
        self.node1 = -1
        self.similarity = 0

# class DataMatrix:
#     def __init__(self, source_name, source_attributes, source_values):
#         self.source_datasource = pd.DataFrame(columns=source_attributes, data=source_values)
#         self.source_name = source_name




# def build_similarity_matrices(data_model, kb_concepts):
#     L = []
#     for key in data_model.datasets:
#         data_instance = data_model.datasets[key]
#         L.append(key)
#         similarity_matrix = (kb_concepts, key)

    # datasource_pairs = itertools.combinations(L, 2)
    # for pair in datasource_pairs:
    #     print(pair)
    #     a = array(pair)


    # for key in data_model.datasets:
    #     data_instance = data_model.datasets[key]
    #     print('Data Instance: ' + key)
    #
    #     for resource in data_instance.resources:
    #         print(resource['format'])
    #
    #         data = resource['data']
    #         first_row = data[0]

from difflib import SequenceMatcher
from similarity.ngram import NGram
twogram = NGram(2)
fourgram = NGram(4)

from similarity.metric_lcs import MetricLCS
metric_lcs = MetricLCS()
def build_local_similarity_matrix(source_schema, target_schema):
    matrix= np.zeros((len(source_schema), len(target_schema)))

    for i in range(len(source_schema)):
            for j in range(len(target_schema)):
                sim_score = 1 - twogram.distance(source_schema[i],target_schema[j])
                # matrix[i,j] = np.int(100*SequenceMatcher(None,source_schema[i],target_schema[j]).ratio())
                matrix[i, j] = sim_score

                if matrix[i, j] >= 0.5:
                    print('matrix[i, j]', source_schema[i], target_schema[j], matrix[i, j])

                # if target_schema[j] == 'tree_species':
                #     print(source_schema[i], target_schema[j], matrix[i, j])

    return matrix


def match_table_by_values_beta(source_instance, target_instance, source_schema, target_schema):
    src_values = []
    tar_values = []

    # src_key = source_instance.keys()[0]
    src_val_len = len(source_instance[source_schema[0]])
    # tar_key = target_instance.keys()[0]
    tar_val_len = len(target_instance[target_schema[0]])

    source_keys = source_schema
        # source_instance.keys()
    # source_keys.sort()
    target_keys = target_schema
        # target_instance.keys()
    # target_keys.sort()

    start_ind = {}
    attr_ind = 0
    val_ind = 0
    for key in source_keys:
        # print('src matrix dimension ' + str(src_val_len) + ' ' + str(len(source_instance[key])))
        # assert src_val_len == len(source_instance[key])
        src_values.extend(source_instance[key])
        for val in source_instance[key]:
            start_ind[val_ind] = attr_ind
            val_ind += 1
        attr_ind =+ 1

    for key in target_keys:
        # print('tar matrix dimension ' + str(src_val_len) + ' ' + str(len(target_instance[key])))
        assert tar_val_len == len(target_instance[key])
        tar_values.extend(target_instance[key])

    sim_matrix = np.zeros((len(source_schema), len(target_schema)))
    for i in range(len(src_values)):
        src_value = src_values[i]
        src_ind = start_ind[i]
        src_attr = source_keys[src_ind]
        for j in range(len(tar_values)):
            tar_value = tar_values[j]
            tar_ind = j // tar_val_len
            tar_attr = target_keys[tar_ind]
            # sim_score = np.int( SequenceMatcher(None, str(src_value), str(tar_value)).ratio())
            sim_score = 1 - twogram.distance(str(src_value), str(tar_value))

            if str(src_value) == 'None' or str(tar_value) == 'None':
                sim_score = 0
            sim_matrix[src_ind, tar_ind] += sim_score

            if sim_score >= 0.5:
                print('sim_score >= 0.5', src_attr, tar_attr, src_value, tar_value, sim_score)

    return sim_matrix

def find_potential_matches(sim_matrix, threshold, src_attrs, tar_attrs, src_name, tar_name):

    matches = []
    for i in range(len(sim_matrix)):
        row = sim_matrix[i]
        max_score = max(row)
        max_indices = [ind for ind, val in enumerate(row) if val == max_score]
        max_rm = [val for ind, val in enumerate(row) if ind not in max_indices]
        matched = all(val < max_score/threshold for val in max_rm)
        matched = matched or max_score >= 0.5 and not (max_score == 0)
        if matched:
            for ind in max_indices:
                matches.append({'i': i, 'j': ind, 'val[i]': src_attrs[i], 'val[j]': tar_attrs[ind], 'src_name': src_name, 'tar_name': tar_name, 'sim_score': sim_matrix[i][ind]})

    return matches

def populate_concept_with_samples(matches_list, dataset, kb_concepts):
    for matches in matches_list:
        for match in matches:
            kb_concept_name = match['val[i]']
            attr_name = match['val[j]']
            datasource_name = match['src_name'][0]

            metadata = dataset[datasource_name][0]
            values = dataset[datasource_name][1]

            # print(kb_concept_name, attr_name, datasource_name)
            concept_source = None
            kb_concept_sources = kb_concepts[kb_concept_name]
            for source in kb_concept_sources:
                if source['source_name'] == datasource_name:
                    concept_source = source

            coded_values = False
            for attr in metadata:
                if attr['name'] == attr_name and attr['domain'] != None and attr['domain'] == 'coded_values':
                    concept_source['coded_values'] = attr['coded_values']
                    coded_values = True

            # use summarization and sampling
            if coded_values == False:
                # DO NOT DO THIS, if no coded values, then do not populate
                # concept_source['coded_values'] = values[attr_name]
                concept_source['coded_values'] = []
            concept_source['sim_score'] = match['sim_score']

import nltk
nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
from nltk.corpus import wordnet
def find_synonyms_antonyms(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                for ant in l.antonyms():
                    antonyms.append(ant.name())
    return (synonyms, antonyms)



if __name__ == "__main__":
    # data_model = parse_dataset.parse_models()
    # graphical_model = build_graph(data_model)
    # graphical_model.cluster()

    # build_similarity_matrices(data_model, kb_concepts)

    # -----
    kb_concepts = parse_dataset.collect_concepts_beta()
    tags = kb_concepts.keys()

    # -----
    import toy.data as td

    # local matches
    PARKS_metadata_tags = list(td.PARKS_metadata['tags'])
    IMPORTANTTREES_metadata_tags = list(td.IMPORTANTTREES_metadata['tags'])
    PARKSPECIMENTREES_metadata_tags = list(td.PARKSPECIMENTREES_metadata['tags'])
    PARKS_metadata_category = [td.PARKS_metadata['category']]
    IMPORTANTTREES_metadata_category = [td.IMPORTANTTREES_metadata['category']]
    PARKSPECIMENTREES_metadata_category = [td.PARKSPECIMENTREES_metadata['category']]
    PARKS_schema_attr = list(td.PARKS_schema.keys())
    IMPORTANTTREES_schema_attr = list(td.IMPORTANTTREES_schema.keys())
    PARKSPECIMENTREES_schema_attr = list(td.PARKSPECIMENTREES_schema.keys())

    schema_sources = [PARKS_metadata_tags,
        IMPORTANTTREES_metadata_tags,
        PARKSPECIMENTREES_metadata_tags,
        PARKS_metadata_category,
        IMPORTANTTREES_metadata_category,
        PARKSPECIMENTREES_metadata_category,
        PARKS_schema_attr,
        IMPORTANTTREES_schema_attr,
        PARKSPECIMENTREES_schema_attr]

    schema_sources_clean = []
    for source in schema_sources:
        schema_source = [val.lower() for val in source]
        schema_source.sort()
        schema_sources_clean.append(schema_source)
        # print(schema_source)

    schema_source_names = [('PARKS', 'tag'), ('IMPORTANTTREES', 'tag'), ('PARKSPECIMENTREES', 'tag'), ('PARKS', 'category'), ('IMPORTANTTREES', 'category'), ('PARKSPECIMENTREES', 'category'), ('PARKS', 'attributes'), ('IMPORTANTTREES', 'attributes'), ('PARKSPECIMENTREES', 'attributes')]

    matching_tasks = [(0, 6), (1, 7), (2, 8), (3, 6), (4, 7), (5, 8)]

    similarity_matrices = []
    for task in matching_tasks:
        similarity_matrices.append(
            (build_local_similarity_matrix(schema_sources_clean[task[0]], schema_sources_clean[task[1]]),
             schema_sources_clean[task[0]],
             schema_sources_clean[task[1]],
             schema_source_names[task[0]],
             schema_source_names[task[1]]))



    matches_list = []
    for matrix in similarity_matrices:
        matches = find_potential_matches(matrix[0], 3, matrix[1], matrix[2], matrix[3], matrix[4])
        matches_list.append(matches)

    print('---matches_list---')
    pp.pprint(matches_list)
    # TODO get all matches, include probabilities

    metadata_files = ['211 Important Trees.json', '239 Park Specimen Trees.json', '244 Parks.json']
    metadata_files = ['./metadata/'+file for file in metadata_files]
    metadata_it = parse_dataset.parse_metadata(metadata_files[0])
    metadata_pst = parse_dataset.parse_metadata(metadata_files[1])
    metadata_parks = parse_dataset.parse_metadata(metadata_files[2])

    # populate each concept with some coded values, based on matches; if no coded values, get a sample of values from instance

    values_it = td.IMPORTANTTREES_data
    values_pst = td.PARKSPECIMENTREES_data
    values_parks = td.PARKS_data

    dataset = {'PARKS': (metadata_parks, values_parks), 'IMPORTANTTREES': (metadata_it, values_it), 'PARKSPECIMENTREES': (metadata_pst, values_pst)}

    populate_concept_with_samples(matches_list, dataset, kb_concepts)

    # print('[kb_concepts]')
    # pp.pprint(kb_concepts)
    # exit(0)

    # print(similarity_matrices[0])
    # print(metadata_parks)
    # # print(schema_sources_clean[0])
    # # print(schema_sources_clean[6])
    # print(matches_list[0])
    # print(values_parks)

    # for concept in kb_concepts:
    #     concept_sources = kb_concepts[concept]
    #     for source in concept_sources:
    #         if 'coded_values' in source:
    #             print(concept, source['source_name'], source['coded_values'], source['sim_score'])




    # match each set of {source attribute values} with each set of {global knowledge base concept values}
    kb_concept_values = {}
    for concept in kb_concepts:
        concept_sources = kb_concepts[concept]
        for source in concept_sources:
            if 'coded_values' in source:
                kb_concept_values[(concept, source['source_name'])] = source['coded_values']

    print('---kb_concept_values---')
    pp.pprint(kb_concept_values)

    # find synonyms for concept
    synonyms_antonyms = {}
    for concept in kb_concept_values.keys():
        if len(kb_concept_values[concept]) < 1:
            continue
        synonyms_antonyms[concept] = find_synonyms_antonyms(concept[0])
        # print('synonyms', concept, ': ', synonyms_antonyms[concept])


    print('---match_table_by_values_beta---')
    for data_source_name in dataset:
        data_source = dataset[data_source_name]
        src = kb_concept_values
        tar = data_source[1]

        src_schema = [key for key in src.keys()]
        tar_schema = [key for key in tar.keys()]
        matrix = match_table_by_values_beta(src, tar, src_schema, tar_schema)
        pp.pprint({'name':data_source_name, 'src':src_schema, 'tar':tar_schema, 'matrix':matrix})

    # summarization:
    # 1) get unique values for an attribute column
    # 2) rank the attribute names by num of non-null values in column

    # training with imperfect labels
    # kb values for concept - probabilities generation and propagation
    # use thesaurus wordnet to find related concepts names, cluster
    # correct knowledge base errors by imperfect training
    # get new concepts from summarization

    # train classifier
    # better distance functions

    # use actual complete datasource values
    # fix potential matrix bug

    # use numpy and pandas

    # create mappings as training data

    # ignore the many-to-one mapping constraint when mapping

    # improve schema matching

    # overcome np-completeness
    # use iterative algorithm
    # use greedy algorithms
    # run in small batches

    # dataset: flat csv or json
    # metadata concepts
    # schema has sample values
    # need some training data!
    # need run on server!

    # measuring precision and recall, need to know real concepts and mappings
        # need to find differences in terms of number of concepts and number of wrong mappings



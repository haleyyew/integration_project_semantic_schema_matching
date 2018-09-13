import os
import sys
import networkx
import itertools

scriptpath = "./parse_dataset.py"
sys.path.append(os.path.abspath(scriptpath))
import parse_dataset

import math
import collections

import sklearn.cluster as cluster
import numpy as np
import matplotlib.pyplot


class GraphicalModel(object):
    def __init__(self):
        self.nodes = []
        self.edges = []

def build_graph(data_model):
    for key in data_model.datasets:
        data_instance = data_model.datasets[key]
        print('Data Instance: ' + key)

        for resource in data_instance.resources:
            print(resource['format'])

            data = resource['data']
            first_row = data[0]
            for attribute_name in first_row:
                print('\t' + attribute_name)
    return

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

if __name__ == "__main__":
    # data_model = parse_dataset.parse_models()
    # build_graph(data_model)

    nodes = {}
    edges = {}

    texts = ['This is a foo bar sentence .',
             'This sentence is similar to a foo bar sentence .',
             'Looks like we have something difference',
             'This sentence looks like not much different than foo bar sentence .']

    resource_names = ['source1', 'source2', 'source1', 'source2']

    texts_vectors = []
    for i in range(len(texts)):
        text = texts[i]
        resource = resource_names[i]
        textvec = word2vec(text)

        node = AttributeNode()
        node.resource_name = resource
        node.attribute_name_vec = textvec
        node.attribute_name = text

        texts_vectors.append(node.id)
        nodes[node.id] = node

    S = complete_graph_from_list(texts_vectors)
    # print(S.edges())
    # print(type(S.edges.keys()))

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

    similarities_array = np.asarray(similarities, dtype=np.float32)

    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(similarities_array.reshape(-1, 1))
    # print(k_means.labels_)

    for j in range(len(k_means.labels_)):
        label = k_means.labels_[j]
        similarity = similarities[j]
        edge = edges[edge_ids[j]]
        print('(', edge.node0, edge.node1, ')', similarity, ':', label)
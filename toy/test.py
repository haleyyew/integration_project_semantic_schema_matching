def test_kb():
    import toy.data as td
    print(td.KNOWLEDGE_BASE['park'])


def test_graph_model():
    pass
    # print(bgm.word2vec('this is a boss'))


def test_similarity():
    from similarity.ngram import NGram
    twogram = NGram(2)
    print(twogram.distance('ABCD', 'ABTUIO'))

    s1 = 'Adobe CreativeSuite 5 Master Collection from cheap 4zp'
    s2 = 'Adobe CreativeSuite 5 Master Collection from cheap d1x'
    fourgram = NGram(4)
    print(fourgram.distance(s1, s2))
    # print(twogram.distance(s1, s2))

    # s2 = 'Adobe CreativeSuite 5 Master Collection from cheap 4zp'
    # print(fourgram.distance(s1, s2))
    #
    # print(fourgram.distance('ABCD', 'ABTUIO'))

    print(1 - fourgram.distance(s1, s2))


def test_numpy_pandas_1():
    import numpy
    x = [[0, 0, 1], [1, 2, 2], [3, 4, 5]]
    y = numpy.array([numpy.array(xi) for xi in x])
    print(y)

    import pandas as pd
    dataframe = pd.DataFrame(data=y, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
    print(dataframe.to_string())

    print('1:', dataframe.columns.get_loc("one"))

    for index, row in dataframe.iterrows():
        print('2:', index, row)
        print()

    print('3:', dataframe.loc['a'])
    print('4:', dataframe.iloc[0])
    print()

    row = dataframe.loc['a']
    headers = list(dataframe.columns.values)
    for i in range(row.size):
        print('5:', headers[i], row[headers[i]])


def test_numpy_pandas_2():
    import numpy as np
    import pandas as pd
    x = [['a', 'b', 'c'], [0, 1, 2], [3, 4, 5]]
    data = np.array([np.array(xi) for xi in x])
    df = pd.DataFrame(data=data[1:, 0:], columns=data[0, 0:])
    print('6:', df.to_string())
    df.to_csv('test_file.csv', sep=',', encoding='utf-8')

    # import os
    # cwd = os.getcwd()
    # print(cwd)

    df2 = pd.read_csv('test_file.csv', index_col=0, header=0)
    print('7:', df2.to_string())


def test_rdf():
    from rdflib import Graph
    g1 = Graph()
    g1.parse("http://bigasterisk.com/foaf.rdf")
    len(g1)



    from rdflib import URIRef, BNode, Literal

    bob = URIRef("http://example.org/people/Bob")
    linda = BNode()

    name = Literal('Bob')
    age = Literal(24)
    height = Literal(76.5)

    # print(bob, linda, name, age, height)

    from rdflib.namespace import RDF, FOAF
    from rdflib import Graph
    g = Graph()

    g.add((bob, RDF.type, FOAF.Person))
    g.add((bob, FOAF.name, name))
    g.add((bob, FOAF.knows, linda))
    g.add((linda, RDF.type, FOAF.Person))
    g.add((linda, FOAF.name, Literal('Linda')))

    g.add((bob, FOAF.age, Literal(42)))
    print("Bob is ", g.value(bob, FOAF.age))

    g.set((bob, FOAF.age, Literal(43)))
    print("Bob is now ", g.value(bob, FOAF.age))

    g.serialize(format='turtle', destination='test_rdf.txt')

    g.remove((bob, None, None))
    g.serialize(format='turtle',destination='test_rdf2.txt')

    g = Graph()
    g.parse("test_rdf.txt", format='turtle')
    import pprint
    for stmt in g:
        pprint.pprint(stmt)

    return

def test_cosine_similarity():
    import re, math
    from collections import Counter

    WORD = re.compile(r'\w+')

    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)

    text1 = 'This is a foo bar sentence .'
    text2 = 'This sentence is similar to a foo bar sentence .'

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)
    print(cosine)

def test_instance_matching():
    import numpy as np
    import pandas as pd
    tar = [['attr1', 'attr2', 'attr3'], ['aaaa', 'bbb', 'ccc'], ['xxx', 'yyyy', 'zzz']]
    # y = [['attr4', 'attr5', 'attr6'], ['xxx', 'yyy', 'zzz'], ['aaa', 'bbb', 'ccc']]
    src = [['attr4'], ['xxx'], ['aaa'], ['mmm']]

    data_tar = np.array([np.array(xi) for xi in tar])
    df_tar = pd.DataFrame(data=data_tar[1:, 0:], columns=data_tar[0, 0:])

    data_src = np.array([np.array(xi) for xi in src])
    df_src = pd.DataFrame(data=data_src[1:, 0:], columns=data_src[0, 0:])

    print(df_tar.to_string())
    print(df_src.to_string())

    schema_tar = list(df_tar.columns.values)
    schema_src = list(df_src.columns.values)

    print(schema_tar)
    print(schema_src)

    src_values = []
    tar_values = []
    src_val_len = 0
    tar_val_len = 0
    for attr in schema_src:
        src_values.extend(list(df_src[attr]))
        src_val_len = len(list(df_src[attr]))

    for attr in schema_tar:
        tar_values.extend(list(df_tar[attr]))
        tar_val_len = len(list(df_tar[attr]))

    from similarity.ngram import NGram
    twogram = NGram(2)

    match_threshold = 0.6
    sim_matrix = np.zeros((len(schema_src), len(schema_tar)))

    for i in range(len(src_values)):
        src_value = src_values[i]
        src_ind = i // src_val_len
        src_attr = schema_src[src_ind]

        for j in range(len(tar_values)):
            tar_value = tar_values[j]
            tar_ind = j // tar_val_len
            tar_attr = schema_tar[tar_ind]

            sim_score = 1 - twogram.distance(str(src_value), str(tar_value))

            if str(src_value) == 'None' or str(tar_value) == 'None':
                sim_score = 0

            if sim_score > match_threshold:
                sim_matrix[src_ind, tar_ind] += sim_score
                print('sim_score >= ', match_threshold, ': ', src_attr, tar_attr, src_value, tar_value, sim_score)

    df_sim_matrix = pd.DataFrame(data=sim_matrix, columns=schema_tar, index=schema_src)
    print(df_sim_matrix.to_string())

def test_groupby():
    import numpy as np
    import pandas as pd
    import pprint
    tar = [['attr1', 'attr2', 'attr3'], ['aaaa', 'bbb', 'ccc'], ['aaaa', 'yyyy', 'zzz'], ['xxx', 'bbb', 'zzz'], ['xxx', str(None), str(None)]]

    data_tar = np.array([np.array(xi) for xi in tar])
    df_tar = pd.DataFrame(data=data_tar[1:, 0:], columns=data_tar[0, 0:])
    print(df_tar.to_string())

    schema_tar = list(df_tar.columns.values)
    kb = {}
    for attr in schema_tar:
        kb[attr] = {}
        groups = df_tar.groupby([attr])[attr]
        print(list(groups.groups.keys()))
        for key, item in groups:
            # print(attr, key, groups.get_group(key).values)
            # print('attr:%s val:%s count:%d' % (attr, key, len(groups.get_group(key).values)))
            kb[attr][key] = len(groups.get_group(key).values)

    pprint.pprint(kb)

def test_compare_datatypes_and_del_cols():
    import build_matching_model as bgm
    import pandas as pd
    import json
    datasets_path = '../thesis_project_dataset_clean/'
    datasource = 'important trees'
    tar_df = pd.read_csv(datasets_path + datasource + '.csv', index_col=0, header=0)
    schema_f = open('../schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)

    tar_df = bgm.df_rename_cols(tar_df)
    tar_schema = list(tar_df.columns.values)

    src_datatype = 'esriFieldTypeString'
    attr_schema = schema_set[datasource]
    cols_to_delete = bgm.compare_datatypes(src_datatype, attr_schema, tar_schema)
    tar_df = bgm.df_delete_cols(tar_df, cols_to_delete)
    print(tar_df.head())

import refine_kb_concepts as rkc

def test_form_new_clusters():


    # matches = {'important trees': {'match_score': 0.58, 'attribute': 'tree species'}, 'park screen trees': {'cluster': {'park specimen trees': {'match_score': 6506, 'attribute': 'tree species'}}, 'match_score': 0.58, 'attribute': 'tree species'}, 'park specimen trees': {'match_score': 0.8, 'attribute': 'tree'}}
    # root = {'trees': matches}

    # TODO change key to datasource.attr b/c there might be multiple probabilistic mappings per datasource
    matches1 = {'ds1': {'match_score': 0.6, 'attribute': 'attr1',
                       'cluster': {'ds3': {'match_score': 1000, 'attribute': 'attr1'}}},
               'ds2': {'match_score': 0.6, 'attribute': 'attr1',
                       'cluster': {'ds3': {'match_score': 1000, 'attribute': 'attr1'}}},
               'ds3': {'match_score': 0.8, 'attribute': 'attr2',
                       'cluster': {'ds1': {'match_score': 2000, 'attribute': 'attr3'},
                                   'ds2': {'match_score': 3000, 'attribute': 'attr4'}
                                   }}
               }
    matches2 = {'ds2': {'match_score': 0.9, 'attribute': 'attr4',
                       'cluster': {'ds3': {'match_score': 3000, 'attribute': 'attr2'},
                                   'ds1': {'match_score': 1800, 'attribute': 'attr3'}
                                   }}
                }
    root = {'c1': matches1, 'c2': matches2}



    # split or merge clusters
    mappings_all = rkc.find_all_subtree_mappings(root, True)
    # pprint.pprint(mappings_all)

    decision_threshold = 0.5
    root = rkc.split_concepts(root, mappings_all, decision_threshold)
    # pprint.pprint(root)

    mappings_all = rkc.find_all_subtree_mappings(root, True)
    # pprint.pprint(mappings_all)

    print('=====')
    decision_threshold = 0.1
    root = rkc.merge_concepts(root, mappings_all, decision_threshold)
    # pprint.pprint(root)
    print('=====')

    return root



def test_find_new_concepts(root):
    import json
    import pprint


    # open output from previous stage
    schema_f = open('../schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)
    datasources = ['important trees', 'park specimen trees', 'park screen trees']
    attr_schema = [schema_set[datasource] for datasource in datasources]

    # add new concepts
    attr_schema_parse = {datasource: [] for datasource in datasources}
    for datasource, schema in zip(datasources, attr_schema):
        for attr in schema:
            name = attr['name']
            attr_schema_parse[datasource].append(name)

    # pprint.pprint(attr_schema_parse)
    # TODO toy example for schema

    attr_schema_parse = {'ds1': ['attr1','attr2','attr3'],
                         'ds2': ['attr1','attr2','attr3','attr4'],
                         'ds3': ['attr1','attr2','attr3','attr4']}

    new_concepts = rkc.create_new_kb_concept(attr_schema_parse, root, True)

    # TODO only select some of these attrs as new concepts in next iteration
    print(new_concepts)
    return new_concepts


def test_hierarchical_cluster_linkage():
    import scipy
    import scipy.cluster.hierarchy as hac
    import scipy.spatial.distance as ssd

    arr = scipy.array([[0, 5], [1, 5], [-1, 5], [0, -5], [1, -5], [-1, -5], [-1.1, -5]])
    pd = ssd.pdist(arr)
    dm = ssd.squareform(pd)
    res1 = hac.linkage(pd)
    res2 = hac.linkage(dm)
    res3 = hac.linkage(arr)

    # print(res1)
    # print(res2)
    # print(dm)
    # print(res3)

    part = hac.fcluster(res3, 0.5, 'inconsistent')
    print(part)

    return

class Paths:
    datasets_path = '../thesis_project_dataset_clean/'
    dataset_metadata_p = '../inputs/datasource_and_tags.json'
    metadata_p = '../inputs/metadata_tag_list_translated.json'
    schema_p = '../inputs/schema_complete_list.json'
    matching_output_p = '../outputs/instance_matching_output/'
    kb_file_p = "../outputs/kb_file.json"
    dataset_stats = '../inputs/dataset_statistics/'
    new_concepts_p = "../outputs/new_concepts.json"
    new_concepts_f = '../outputs/new_concepts.csv'

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

p = Paths()

def test_new_concepts_from_sem():
    import json
    import iterative_algorithm as ia

    input_topics = ['trees', 'parks']
    input_datasets = []
    kb = {}

    dataset_metadata_set, metadata_set, schema_set, datasources_with_tag = ia.load_metadata(p, input_topics,
                                                                                            input_datasets)
    _, datasources_with_tag, datasources_index, reverse_index = ia.find_datasources(p, datasources_with_tag, input_topics, kb)

    kb_f = open(p.kb_file_p, 'r')
    kb = json.load(kb_f)

    num_of_new_concepts = 5
    new_concepts, new_concepts_mod, concept_sims_scores, _ = ia.find_new_concepts(p, metadata_set, schema_set, kb, datasources_index, num_of_new_concepts, input_topics)

    return

def test_prepare_next_iteration():
    import pandas as pd
    import json
    import iterative_algorithm as ia
    import build_matching_model as bmm

    kb = {}

    num = 5
    new_concepts = pd.read_csv(p.new_concepts_f, index_col=0, header=0)
    top_concepts = new_concepts.sort_values('score', ascending=False).head(num)
    output_new_concepts = list(top_concepts['concept'])

    print(output_new_concepts)

    kb_f = open(p.kb_file_p, 'r')
    kb = json.load(kb_f)
    print(kb.keys())

    input_topics = output_new_concepts

    dataset_metadata_set, metadata_set, schema_set, datasources_with_tag = ia.load_metadata(p, input_topics, None)
    print(datasources_with_tag)

    kb, datasources_with_tag, datasources_index, reverse_index = ia.find_datasources(p, datasources_with_tag, input_topics, kb)
    print(kb.keys())

    bmm.gather_statistics(schema_set, datasources_with_tag, p.dataset_stats, p.datasets_path)

    kb, datasources_with_tag, schema_set = ia.initialize_matching(p, input_topics, dataset_metadata_set, schema_set,
                                                               datasources_with_tag, reverse_index, kb)
    with open(p.schema_p, 'w') as fp:
        json.dump(schema_set, fp, sort_keys=True, indent=2)

    kb_file = open(p.kb_file_p, "w")
    json.dump(kb, kb_file, indent=2, sort_keys=True)

    return

test_prepare_next_iteration()
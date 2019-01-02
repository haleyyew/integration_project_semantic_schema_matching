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

    print(dataframe.columns.get_loc("one"))

    for index, row in dataframe.iterrows():
        print(index, row)
        print()

    print(dataframe.loc['a'])
    print(dataframe.iloc[0])
    print()

    row = dataframe.loc['a']
    headers = list(dataframe.columns.values)
    for i in range(row.size):
        print(headers[i], row[headers[i]])


def test_numpy_pandas_2():
    import numpy as np
    import pandas as pd
    x = [['a', 'b', 'c'], [0, 1, 2], [3, 4, 5]]
    data = np.array([np.array(xi) for xi in x])
    df = pd.DataFrame(data=data[1:, 0:], columns=data[0, 0:])
    print(df.to_string())
    df.to_csv('test_file.csv', sep=',', encoding='utf-8')

    # import os
    # cwd = os.getcwd()
    # print(cwd)

    df2 = pd.read_csv('test_file.csv', index_col=0, header=0)
    print(df2.to_string())


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

test_groupby()
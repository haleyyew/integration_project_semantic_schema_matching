
def test_kb():
    import toy.data as td
    print(td.KNOWLEDGE_BASE['park'])

def test_graph_model():
    import build_graphical_model as bgm
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

    print(1-fourgram.distance(s1, s2))

def test_numpy_pandas_1():
    import numpy
    x = [[0,0,1],[1,2,2],[3,4,5]]
    y=numpy.array([numpy.array(xi) for xi in x])
    print(y)

    import pandas as pd
    dataframe = pd.DataFrame(data=y, columns=['one','two','three'], index=['a','b','c'])
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
    x = [['a','b','c'],[0,1,2],[3,4,5]]
    data = np.array([np.array(xi) for xi in x])
    df = pd.DataFrame(data=data[1:,0:], columns=data[0,0:])
    print(df.to_string())
    df.to_csv('test_file.csv', sep=',', encoding='utf-8')

    # import os
    # cwd = os.getcwd()
    # print(cwd)

    df2 = pd.read_csv('test_file.csv', index_col=0, header=0)
    print(df2.to_string())


test_numpy_pandas_2()
DEBUG_MODE = False

from similarity.ngram import NGram
twogram = NGram(2)
def matcher_name(src, tar, function):
    sim_score = 1 - function.distance(src, tar)
    return sim_score


import nltk

def load_dict():
    nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
    from nltk.corpus import wordnet
    dictionary = wordnet
    return dictionary

import pandas as pd
import inflection
import math
def matcher_name_meaning_by_thesaurus(src, tar, dictionary):

    # threshold = 0.2
    # top_rows = 0.05
    threshold = 0.5
    top_rows = 0.4

    src_word_vec = src.split(' ')
    tar_word_vec = tar.split(' ')

    src_word_enrich = {word: {} for word in src_word_vec}
    tar_word_enrich = {word: {} for word in tar_word_vec}

    for word1 in tar_word_vec:
        word1 = inflection.singularize(word1)

        w1 = None
        try:
            w1 = dictionary.synsets(word1, pos=dictionary.NOUN)
            tar_word_enrich[word1] = w1
        except Exception:
            continue

    for word2 in src_word_vec:
        word2 = inflection.singularize(word2)

        w2 = None
        try:
            w2 = dictionary.synsets(word2, pos=dictionary.NOUN)
            src_word_enrich[word2] = w2
        except Exception:
            continue

    # sem_sim_score = 0
    sims_list = []
    for word1 in tar_word_vec:
        for word2 in src_word_vec:
            if len(tar_word_enrich[word1]) == 0 or len(src_word_enrich[word2]) == 0:
                continue

            # synonym approach 1
            # intersect = set(tar_word_enrich[word1].lemma_names).intersection(set(src_word_enrich[word2].lemma_names))
            # print(word1, tar_word_enrich[word1], word2, src_word_enrich[word2], intersect)

            # synonym approach 2
            sims = find_similarity_between_synset_pairs(tar_word_enrich[word1], src_word_enrich[word2], dictionary)
            # print(sims.to_string())

            num_pairs = len(sims.index)
            sims = sims.head(max(int(math.ceil(num_pairs * top_rows)), 1))
            # sims = sims.head(1)

            append_pair = False
            for k, v in sims.iterrows():
                lemmas_1 = show_all_lemmas(v['synset_1'], [])
                lemmas_2 = show_all_lemmas(v['synset_2'], [])

                hyperset_1 = set([i for i in v['synset_1'].closure(lambda s:s.hypernyms())])
                hyperset_2 = set([i for i in v['synset_2'].closure(lambda s: s.hypernyms())])

                if DEBUG_MODE:
                    print(lemmas_1, lemmas_2)
                    print(v['synset_2'] in hyperset_1)
                    print(v['synset_1'] in hyperset_2)

                if v['sim'] > threshold:
                    append_pair = True

            if append_pair == True:
                sims_list.append((word1, word2, sims))

    ## compute score method 1
    #         sem_sim = w1.wup_similarity(w2)
    #         sem_sim_score += sem_sim
    #         # print(w1, w2, sem_sim)
    #
    # sem_sim_score = sem_sim_score / (len(src_word_vec) * len(tar_word_vec))
    # return sem_sim_score

    if len(sims_list) == 0:
        return 0, []

    scores = []
    for sims_tuple in sims_list:
        sims = sims_tuple[2]
        for k, v in sims.iterrows():
            sim_score = v['sim']
            scores.append(sim_score)

    return max(scores), sims_list

def find_similarity_between_synset_pairs(synsets_1, synsets_2, wn):
    df = pd.DataFrame(columns=['sim', 'synset_1', 'synset_2'])

    for synset_1 in synsets_1:
        for synset_2 in synsets_2:
            sim = wn.path_similarity(synset_1, synset_2)
            # sim = w1.wup_similarity(w2)
            if sim is not None:

                df = df.append({'sim': sim, 'synset_1': synset_1, 'synset_2': synset_2}, ignore_index=True)

    df = df.sort_values(by=['sim'], ascending=False)
    return df

def show_all_lemmas(synset, exclude):

    lemmas = []
    lemmas += [str(lemma.name()) for lemma in synset.lemmas()]
    lemmas = [synonym.replace("_", " ") for synonym in lemmas]
    lemmas = list(set(lemmas))
    lemmas = [synonym for synonym in lemmas if synonym not in exclude]
    return lemmas


def matcher_data_type(type1, type2):
    if str(type1) == str(type2):
        return 1.0
    else:
        return 0.0

from scipy import spatial
import re
def matcher_instance_document(text1, text2):

    text1 = rm_special_chars(text1)
    text2 = rm_special_chars(text2)

    tf_dict1, word_freq_dict1, text_vec1, doc_len1 = compute_tf(text1)
    tf_dict2, word_freq_dict2, text_vec2, doc_len2 = compute_tf(text2)

    factor = 2
    tf_dict1, tf_dict2 = compute_df(tf_dict1, tf_dict2, factor)

    vec1, vec2, all_words = compute_freq_vec(tf_dict1, tf_dict2)
    if DEBUG_MODE: print(vec1, vec2, all_words)

    return 1 - spatial.distance.cosine(vec1, vec2)

def rm_special_chars(s):
    s = re.sub('[^\w\s]', '', s)
    s = re.sub('_', '', s)
    s = re.sub('\s+', ' ', s)
    s = s.strip()

    s = s.split()
    s = [word.lower() for word in s]
    s = ' '.join(s)
    return s

def compute_tf(s):
    word_freq_dict = {}

    text_vec = s.split()
    text_vec = [word.lower() for word in text_vec]
    doc_len = len(text_vec)

    for word in text_vec:
        if word in word_freq_dict:
            word_freq_dict[word] += 1
        else:
            word_freq_dict[word] = 1

    tf_dict = {}
    for word in word_freq_dict:
        tf_dict[word] = word_freq_dict[word] / doc_len

    return tf_dict, word_freq_dict, text_vec, doc_len

def compute_df(freq1, freq2, factor):
    words1 = list(freq1.keys())
    words2 = list(freq2.keys())

    if len(words1) > len(words2):
        words = words2
    else:
        words = words1

    # reward for word appearing in both vectors
    for word in words:
        if word in words1 and word in words2:
            freq1[word] = freq1[word] * factor
            freq2[word] = freq2[word] * factor

    return freq1, freq2

def compute_freq_vec(tf_dict1, tf_dict2):
    all_words = list(set(tf_dict1.keys()).union(set(tf_dict2.keys())))
    all_words.sort()
    all_words_len = len(all_words)

    vec1 = [0] * all_words_len
    vec2 = [0] * all_words_len

    for i in range(all_words_len):
        word = all_words[i]
        if word in tf_dict1:
            vec1[i] = tf_dict1[word]
        if word in tf_dict2:
            vec2[i] = tf_dict2[word]

    return vec1, vec2, all_words


from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
def matcher_attribute_descriptions(path, text1, text2):
    # using gensim lda

    temp_file = datapath(path + 'lda_model')
    lda = LdaModel.load(temp_file)

    dictionary = Dictionary.load(path + 'dict')
    # common_dictionary = Dictionary(common_texts)
    # print(lda.print_topics(5))

    text1 = rm_special_chars(text1)
    text2 = rm_special_chars(text2)

    text1 = text1.split()
    text2 = text2.split()

    corpus = [text1, text2]
    # print(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    # print(corpus)

    vector1 = lda[corpus[0]]
    vector2 = lda[corpus[1]]

    from pprint import pprint
    # pprint(vector1)
    # pprint(vector2)
    vector1 = sorted(vector1, key=lambda x: x[1], reverse=True)
    vector2 = sorted(vector2, key=lambda x: x[1], reverse=True)
    print(vector1)
    print(vector2)

    topics1 = [(dictionary[tup[0]], tup[1]) for tup in vector1]
    topics2 = [(dictionary[tup[0]], tup[1]) for tup in vector2]
    print(topics1)
    print(topics2)

    return

def train_lda(path):
    # from gensim.test.utils import common_texts
    # from gensim.corpora.dictionary import Dictionary
    # from gensim.models.ldamodel import LdaModel
    # from gensim.test.utils import datapath

    # common_dictionary = Dictionary(common_texts)
    # common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

    # print(common_dictionary.get(80))

    # lda = LdaModel(common_corpus, num_topics=5)
    # temp_file = datapath(path)
    # lda.save(temp_file)
    # lda = LdaModel.load(temp_file)


    documents = ["Amazon sells many things " ,
                 "Apple is releasing a new product " ,
                 "Microsoft announces Nokia acquisition " ,
                 'Julie loves me more than Linda loves me ' ,
                 'Jane likes me more than Julie loves me']

    documents = [rm_special_chars(s) for s in documents]
    stoplist = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', "don't", 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']


    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    # print(texts)
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # print(dictionary.get(0))

    num_topics = 3
    lda = LdaModel(corpus=corpus, id2word=dictionary,
                   num_topics=num_topics, update_every=1,
                   random_state=100, chunksize=100,
                   passes=20, alpha='auto')

    temp_file = datapath(path + 'lda_model')
    lda.save(temp_file)
    lda = LdaModel.load(temp_file)

    dictionary.save(path + 'dict')

    return lda, dictionary

def matcher_by_hmm():
    return

# def matcher_by_naive_bayes():
#     return

def matcher_by_semantic_hash():
    # train neural net
    return

# score = matcher_name('park', 'green', twogram)
# dictionary = load_dict()
# score, sims_list = matcher_name_meaning_by_thesaurus('park', 'green', dictionary)
# score = matcher_instance_document('Julie loves me more than Linda loves me', 'Jane likes me more than Julie loves me')
# train_lda('/Users/haoran/Documents/thesis_schema_integration/outputs/stat_models/')
# # TODO does not work well
# matcher_attribute_descriptions(
#     '/Users/haoran/Documents/thesis_schema_integration/outputs/stat_models/',
#     'Julie loves me more than Linda loves me',
#     'Jane likes me more than Julie loves me')
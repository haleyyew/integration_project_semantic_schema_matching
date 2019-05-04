import nltk

DEBUG_MODE = 0

from similarity.ngram import NGram
twogram = NGram(2)
def matcher_name(src, tar, function):
    sim_score = 1 - function.distance(src, tar)
    return sim_score


def load_dict():
    nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
    from nltk.corpus import wordnet
    dictionary = wordnet
    return dictionary

import pandas as pd
import inflection
import math
def matcher_name_meaning_by_thesaurus(src, tar, dictionary, threshold):

    # threshold = 0.2
    # top_rows = 0.05

    top_rows = 1.0

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


    for word1 in tar_word_vec:
        for word2 in src_word_vec:
            if len(tar_word_enrich[word1]) == 0 or len(src_word_enrich[word2]) == 0:
                sim2 = matcher_name(word1, word2, twogram)
                # if 'businesses' in word1 or 'businesses' in word2:
                #     print('===', word1, word2, ' : ', sim2)

                if sim2 > threshold*2:
                    df = pd.DataFrame(columns=['sim', 'word1', 'word2'])
                    df = df.append({'sim': sim2, 'word1': word1, 'word2': word2}, ignore_index=True)
                    sims_list.append((word1, word2, df))


    ## compute score method 1
    #         sem_sim = w1.wup_similarity(w2)
    #         sem_sim_score += sem_sim
    #         # print(w1, w2, sem_sim)
    #
    # sem_sim_score = sem_sim_score / (len(src_word_vec) * len(tar_word_vec))
    # return sem_sim_score

    if len(sims_list) == 0:
        # print('empty: ', src, tar)
        return 0, []

    scores = []
    for sims_tuple in sims_list:
        sims = sims_tuple[2]
        for k, v in sims.iterrows():

            # word1 = sims_tuple[0]
            # word2 = sims_tuple[1]
            # sim2 = matcher_name(word1, word2, twogram)
            # if 'businesses' in word1 or 'businesses' in word2:
            #     print(word1, word2, ' : ', v['sim'], sim2)
            # v['sim'] = max(v['sim'], sim2)

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


import pprint
import numpy as np
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
import scipy

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


import pickle
import json
import numpy as np
def cluster_topics_prep():

    dictionary = load_dict()

    metadata_f = open('./inputs/metadata_tag_list_translated.json', 'r')
    metadata_set = json.load(metadata_f)

    # print(len(metadata_set))

    total_topics_num = len(metadata_set['groups']) + len(metadata_set['tags'])

    # print(total_topics_num)
    # print(metadata_set['groups'].keys())

    groups = list(metadata_set['groups'].keys())
    tags = list(metadata_set['tags'].keys())

    groups.sort()
    tags.sort()

    # print(groups[:5])
    # print(tags[:5])

    index_of = 0
    topics_dict = {}
    duplicate_list = []
    topics_new = groups.copy()
    for group in groups:
        if group in tags:
            # print('error', group)
            duplicate_list.append(group)
            topics_new.remove(group)
            continue
        topics_dict[group] = index_of
        index_of += 1
    # print(groups_new)
    # print(duplicate_list)

    for tag in tags:
        topics_dict[tag] = index_of
        index_of += 1
    topics_new.extend(tags)


    # index_of = 5
    # topics_new = ['business', 'businesses', 'finance and commerce', 'startup business', 'establishmnt']


    # print(index_of)
    sim_matrix = np.zeros(shape=(index_of,index_of))
    # print(sim_matrix.shape)

    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'

    num_topics = len(topics_new)
    row_index = 0
    for topic in topics_new:


        # if row_index <= 274:
        #     row_index += 1
        #     continue


        # col_index = 0
        for col_index in range(row_index, num_topics, 1):
            target = topics_new[col_index]
            # topic_vec = topic.split()
            # target_vec = target.split()
            # if len(topic_vec) > 1:
            #     print(topic_vec)
            # if len(target_vec) > 1:
            #     print(target_vec)
            threshold = 0.34
            score, sims_list = matcher_name_meaning_by_thesaurus(topic, target, dictionary, threshold)
            if score > threshold:
                print(topic, '<=>', target, ' : ', score)
            sim_matrix[row_index][col_index] = score
            # col_index += 1

        np.savetxt(dir + "topic_sims"+str(row_index)+".csv", sim_matrix, delimiter=",")
        print('done topic #', row_index)
        row_index += 1

    # sim_matrix= np.asarray(sim_matrix)
    np.savetxt(dir+"topic_sims.csv", sim_matrix, delimiter=",")

    with open(dir+'topics.txt', 'wb') as fp:
        pickle.dump(topics_new, fp)

    return

def cluster_topics_prep_matrix():
    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    sim_matrix = np.loadtxt(dir+"topic_sims756.csv", delimiter=',')
    # sim_matrix_old = np.loadtxt(dir + "topic_sims274.csv", delimiter=',')

    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    # for i in range(274+1):
    #     for j in range(len(topics_new)):
    #         sim_matrix[i][j] = sim_matrix_old[i][j]

    for i in range(len(topics_new)):
        for j in range(len(topics_new)):
            sim_matrix[i][j] = round(sim_matrix[i][j], 4)
            sim_matrix[j][i] = sim_matrix[i][j]

    np.savetxt(dir + "topic_sims.csv", sim_matrix, delimiter=",")
    print('done saving')



    return

def cluster_topics():
    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    sim_matrix = np.loadtxt(dir+"topic_sims.csv", delimiter=',')

    decision_threshold = 0.45
    part = hierarchical_cluster_linkage(sim_matrix, decision_threshold)
    pprint.pprint(part)

    part_indexes = [[part[i], i] for i in range(len(part))]
    part_indexes_df = pd.DataFrame(part_indexes, columns=['group', 'index'])
    groups_df = part_indexes_df.groupby('group')['index'].apply(list)

    print(groups_df.head())

    groups_df.to_csv(dir + 'topics_groups.csv', sep=',', encoding='utf-8', index=False)

    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    reverse_topic_cluster = {topics_new[item]: index for index, row in groups_df.iteritems() for item in row}
    topic_cluster = {index: list(row) for index, row in groups_df.iteritems()}

    # print(reverse_topic_cluster)

    return groups_df, reverse_topic_cluster, topic_cluster

def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def open_updated_topics(dir, source_name):
    updated_topics_file = dir + 'new_topics_[' + source_name + '].txt'
    if os.path.isfile(updated_topics_file):
        with open(updated_topics_file, 'rb') as fp:
            updated_topics_this = pickle.load(fp)

            return True, updated_topics_this

    return False, None

import textwrap
def print_datasets_with_topic(dataset_metadata_set, dataset_with_topic, dir):
    print('     dataset: ', dataset_with_topic)
    dataset_existing_tags = dataset_metadata_set[dataset_with_topic]['tags']
    dataset_existing_groups = dataset_metadata_set[dataset_with_topic]['groups']
    dataset_notes = dataset_metadata_set[dataset_with_topic]['notes']

    dataset_existing_tags = [tag['display_name'] for tag in dataset_existing_tags]
    dataset_existing_groups = [group['display_name'] for group in dataset_existing_groups]
    dataset_notes = [word for word in dataset_notes.split() if "http://" not in word]



    print('     tags: ', textwrap.fill(str(dataset_existing_tags), 120))
    print('     groups: ', textwrap.fill(str(dataset_existing_groups), 120))
    print('     notes: ', textwrap.fill(' '.join(dataset_notes), 120))

    # get more from updated list
    update, updated_topics_this = open_updated_topics(dir, dataset_with_topic)
    if update: print(updated_topics_this)
    print('--')

    return

def input_from_command(add_list, delete_list, topics_new):
    option = input("==add topics==")
    adding = option.split(',')
    # print(adding)
    adding = [topics_new[int(num)] for num in adding if isint(num)]
    add_list.extend(adding)
    option = input("==del topics==")
    deleting = option.split(',')
    deleting = [topics_new[int(num)] for num in deleting if isint(num)]
    delete_list.extend(deleting)

    return add_list, delete_list


import parse_dataset as pds
import build_matching_model as bmm
import iterative_algorithm as ia
import os.path
def recommend_labels():
    import json
    import pandas as pd

    dataset_metadata_f = open('./inputs/datasource_and_tags.json', 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open('./inputs/metadata_tag_list_translated.json', 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open('./inputs/schema_complete_list.json', 'r')
    schema_set = json.load(schema_f, strict=False)

    group = 'environmental services'

    datasources_with_tag = metadata_set['groups'][group]['sources']
    datasets_path = './thesis_project_dataset_clean/'

    print(datasources_with_tag)
    datasources_with_tag = [datasource_file for datasource_file in datasources_with_tag if os.path.isfile(datasets_path+datasource_file+'.csv') ]
    print(datasources_with_tag)

    p = ia.p
    bmm.gather_statistics(schema_set, datasources_with_tag, p.dataset_stats, p.datasets_path)

    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    sim_matrix = np.loadtxt(dir + "topic_sims.csv", delimiter=',')
    clusters_of_concepts, reverse_topic_cluster, topic_cluster = cluster_topics()


    topic_similarities = {}
    for topic in topics_new:
        sims = sim_matrix[topics_new.index(topic)]
        non_zero = [(i,topics_new[i]) for i, e in enumerate(sims) if e != 0]
        topic_similarities[topic] = non_zero


    for source_name in datasources_with_tag:
        print('-----', source_name, '-----')
        dataset = pd.read_csv(datasets_path + source_name + '.csv', index_col=0, header=0)

        data_samples = dataset.head()
        col_names = schema_set[source_name]
        dataset_existing_tags = dataset_metadata_set[source_name]['tags']
        dataset_existing_groups = dataset_metadata_set[source_name]['groups']
        dataset_notes = dataset_metadata_set[source_name]['notes']

        # display all datasets for each topic
        datasets_with_tag = {}
        for tag in dataset_existing_tags:
            tag_name = tag['display_name']
            print(tag_name)
            # print(metadata_set['tags'][tag_name])
            datasets_with_tag[tag_name] = metadata_set['tags'][tag_name]['sources']

        # get more topics from updated list
        updated_topics_this = open_updated_topics(dir, source_name)
        print(updated_topics_this)

        # allow break out from current source
        option = input("==skip this dataset? (y)==")
        if option == "y":
            continue

        add_list = []
        # recommend topics close to dataset
        for col in col_names:
            if col['alias'] != None:
                col_name = col['alias']
            else:
                col_name = col['name']

            threshold = 0.5
            for topic in topics_new:
                score = matcher_name(col_name, topic, twogram)
                if score > threshold:
                    print(topic, "<=>", col_name, score)
                    print("similar topics")
                    print(topic_similarities[topic])
                    print("topic cluster")
                    cluster_for_topic = topic_cluster[reverse_topic_cluster[topic]]
                    # topic ids to name
                    print([topics_new[id] for id in cluster_for_topic])
                    print("datasets with topic")
                    if topic in metadata_set['tags']:
                        print(metadata_set['tags'][topic]['sources'])
                    elif topic in metadata_set['groups']:
                        print(metadata_set['groups'][topic]['sources'])
                    else:
                        print("error: topic cannot be found")
                    print("data samples")
                    print(data_samples)
                    print("data description")
                    print(dataset_notes)
                    print("data topics")
                    print(dataset_existing_tags)
                    print("data groups")
                    print(dataset_existing_groups)
                    print("-----")

                    option = input("==accept or reject (y)==")
                    if option == "a" or option == "y":
                        print("accept", topic)
                        add_list.append(topic)
                    else:
                        print("reject", topic)

            print('----------')
            print()
            print()
            print()
        print('==========')
        print()
        print()
        print()
        print()
        print()

        delete_list = []
        for topic in dataset_existing_tags:
            print("[",topic['display_name'],"]")
            similar_topics = topic_similarities[topic['display_name']]
            print("similar topics:")

            # show all datasets with topic, and additional topics each dataset has

            for topic_sim in similar_topics:
                print('=    ', topic_sim)
                topic_sim_name = topic_sim[1]
                datasets_with_topic = None

                if topic_sim_name in metadata_set['groups']:
                    # print('found in groups')
                    datasets_with_topic = metadata_set['groups'][topic_sim_name]['sources']

                if topic_sim_name in metadata_set['tags']:
                    # print('found in tags')
                    datasets_with_topic = metadata_set['tags'][topic_sim_name]['sources']

                if topic_sim_name in metadata_set['groups'] or topic_sim_name in metadata_set['tags']:
                    count_sim_topics_printed = 0

                    for dataset_with_topic in datasets_with_topic:
                        count_sim_topics_printed += 1
                        print_datasets_with_topic(dataset_metadata_set, dataset_with_topic, dir)
                        if count_sim_topics_printed % 5 == 0:
                            add_list, delete_list = input_from_command(add_list, delete_list, topics_new)

                # print('===')

                add_list, delete_list = input_from_command(add_list, delete_list, topics_new)

                print("-add_list-", add_list)
                print("-delete_list-", delete_list)

        while True:
            print("==add additional topics or delete existing topics==")
            add_list, delete_list = input_from_command(add_list, delete_list, topics_new)

            option = input("done? (e)")
            if option == "e":
                break

            continue

        print("-add_list-", add_list)
        print("-delete_list-", delete_list)


        print("[save]")

        existing_tags = [item['display_name'] for item in dataset_existing_tags]
        existing_groups = [item['display_name'] for item in dataset_existing_groups]

        print("-existing data topics-", existing_tags)
        print("-existing data groups-", existing_groups)

        new_topics_set = set([*add_list, *existing_tags, *existing_groups])

        for item in delete_list:
            new_topics_set.remove(item)
        print(new_topics_set)

        with open(dir + 'new_topics_['+source_name+'].txt', 'wb') as fp:
            pickle.dump(new_topics_set, fp)

    return


# cluster_topics_prep_matrix()
# cluster_topics()
recommend_labels()
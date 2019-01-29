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
        w1 = wordnet.synset(word1+'.n.01')
        for word2 in src_word_vec:
            word2 = inflection.singularize(word2)
            w2 = wordnet.synset(word2 + '.n.01')
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
        return False, sem_sims
    else:
        # leave concept, create new concept
        return True, sem_sims

def create_new_kb_concept(kb):
    return

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

    # TODO change to examine mapping clusters
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
            leave_concept, sem_sims = find_similarity_between_wordsets(concept, src_attr, cluster_attrs)
            print(leave_concept, sem_sims)

            if leave_concept:
                create_new_kb_concept(kb)

    # TODO look at all datasources, pick some attributes not covered in kb as new concepts

    kb_file = open("kb_file.json", "w")
    json.dump(kb, kb_file, indent=2, sort_keys=True)


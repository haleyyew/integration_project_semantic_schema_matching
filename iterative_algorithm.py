DEBUG_MODE = False

import json
import csv
import numpy as np
import pandas as pd
import time
import os
import sys
from pathlib import Path
import pprint

from numpy import array
import inflection
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
import scipy

import nltk
import platform
pltfm = platform.system()
if pltfm == 'Linux':
    nltk.data.path.append('/home/haoran/Documents/venv/nltk_data/')
else:
    nltk.data.path.append('/Users/haoran/Documents/nltk_data/')
from nltk.corpus import wordnet

from similarity.ngram import NGram
twogram = NGram(2)
fourgram = NGram(4)

import parse_dataset as pds
import build_matching_model as bmm
import refine_kb_concepts as rkc

datasets_path = './thesis_project_dataset_clean/'
dataset_metadata_p = './inputs/datasource_and_tags.json'
metadata_p = './inputs/metadata_tag_list_translated.json'
schema_p = './inputs/schema_complete_list.json'
matching_output_p = './outputs/instance_matching_output/'
kb_file_p = "./outputs/kb_file.json"
dataset_stats = './inputs/dataset_statistics/'
new_concepts_p = "./outputs/new_concepts.json"

def load_metadata(input_topics, input_datasets):
    '''TODO there might be a correct mapping between input_topics and attributes of input_datasets'''

    dataset_metadata_f = open(dataset_metadata_p, 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open(metadata_p, 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open(schema_p, 'r')
    schema_set = json.load(schema_f, strict=False)

    datasources_with_tag = {}
    for topic in input_topics:
        datasources_with_tag[topic]= metadata_set['tags'][topic]['sources']

    dataset_metadata_f.close()
    metadata_f.close()
    schema_f.close()

    return dataset_metadata_set, metadata_set, schema_set, datasources_with_tag


def find_datasources(datasources_with_tag):
    kb = {}

    reverse_index = {}
    for topic in input_topics:
        for data_src in datasources_with_tag[topic]:
            path = datasets_path + data_src + '.csv'
            my_file = Path(path)
            if not my_file.exists():
                datasources_with_tag[topic].remove(data_src)
                continue

            if data_src not in reverse_index:
                reverse_index[data_src] = [topic]
            else:
                reverse_index[data_src].append(topic)

        kb = bmm.build_kb_json([(topic, datasources_with_tag[topic])], kb)

    datasources_index = datasources_with_tag
    datasources_with_tag = list(reverse_index.keys())
    return kb, datasources_with_tag, datasources_index, reverse_index

def initialize_matching(input_topics, dataset_metadata_set, schema_set, datasources_with_tag, reverse_index, kb):

    datasources = {}
    for source_name in datasources_with_tag:
        # path = datasets_path + source_name + '.csv'
        # dataset = pd.read_csv(path, index_col=0, header=0)

        schema = schema_set[source_name]
        metadata = dataset_metadata_set[source_name]['tags']
        dataset = None

        # dataset = bmm.df_rename_cols(dataset)

        datasources[source_name] = (source_name, dataset, schema, metadata)

        print(source_name)
        if DEBUG_MODE: bmm.print_metadata_head(source_name, dataset, schema, metadata)

        # initialization schema matching
        tags_list = [tag['display_name'] for tag in metadata]
        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
        sim_matrix = bmm.build_local_similarity_matrix(tags_list, attributes_list)
        sim_frame = pd.DataFrame(data=sim_matrix, columns=attributes_list, index=tags_list)

        # print(sim_frame.to_string())

        # TODO during new concepts stage, add second best tag and so on
        attrs = list(sim_frame.columns.values)
        for topic in reverse_index[source_name]:

            max_score = 0
            arg_max_score = None
            arg_i = -1
            for attr_i in range(len(attrs)):
                attr = attrs[attr_i]
                score = sim_frame.loc[topic, attr]
                if score > max_score:
                    max_score = score
                    arg_max_score = attr
                    arg_i = attr_i

            arg_max_examples_vals = None
            example_value = None
            if schema[arg_i]['domain'] != None:
                arg_max_examples_vals = schema[arg_i]['coded_values']
                arg_max_examples_vals.sort()
                example_value = arg_max_examples_vals[0]
            else:
                # TODO replace with loading from stats file
                _, uniques = bmm.get_attr_stats(dataset_stats, source_name, attrs[arg_i])
                # stat, _, uniques = bmm.groupby_unique(attrs[arg_i], dataset)

                uniques.sort()
                schema[arg_i]['coded_values'] = uniques
                arg_max_examples_vals = schema[arg_i]['coded_values']

                print('arg_max_examples_vals', arg_max_examples_vals[0])

                schema[arg_i]['domain'] = 'coded_values_groupby'

            print('best match:', topic, arg_max_score, max_score, example_value)

            kb_match_entry = {'concept': topic,
                              'datasource': source_name,
                              'attribute': arg_max_score,
                              'match_score': max_score,
                              'example_values': arg_max_examples_vals,
                              'data_type': schema[arg_i]['data_type']}

            bmm.update_kb_json(kb, kb_match_entry)
        print('-----')

    # done initialization

    return kb, datasources_with_tag, schema_set

def perform_matching(dataset_metadata_set, schema_set, datasources_with_tag, kb, params):
    comparison_count = 0
    comparison_count_o = [comparison_count]
    sim_matrices = {}

    for source_name in datasources_with_tag:
        t2 = time.time()

        dataset = pd.read_csv(datasets_path + source_name + '.csv', index_col=0, header=0)
        dataset = bmm.df_rename_cols(dataset)

        schema = schema_set[source_name]
        metadata = dataset_metadata_set[source_name]['tags']
        schema = [{'name': pds.clean_name(attr['name'], False, False)} for attr in schema]

        schema_attr_names = []
        for attr in schema:
            # attr['name'] = pds.clean_name(attr['name'], False, False)
            schema_attr_names.append(attr['name'])
        schema_attr_names.sort()

        for concept in kb:
            for datasource in kb[concept]['matches']:
                src_attr = kb[concept]['matches'][datasource]['attribute']
                src_vals = kb[concept]['matches'][datasource]['example_values']
                # do not match with self
                if source_name == datasource:
                    continue
                # do not match if no populated values
                if src_vals == None:
                    continue

                src_data = pd.DataFrame({src_attr: src_vals})
                print("[concept:%s, datasource:%s(%s) <=> dataset:%s]" % (concept, datasource, src_attr, source_name))

                # groupby values for each column and obtain count for each unique value, then multiply counts when comparison succeeds
                tar_schema = list(dataset.columns.values)
                cols_to_delete = bmm.find_attrs_to_delete(schema, tar_schema)
                tar_schema = [item for item in tar_schema if item not in cols_to_delete]

                attrs_stat = {}
                max_len = 0
                for attr in tar_schema:
                    # TODO save this output to file for later use
                    # stat, groups, uniques = bmm.groupby_unique(attr, dataset)
                    stat, uniques = bmm.get_attr_stats(dataset_stats, source_name, attr)
                    uniques.sort()

                    # save for later
                    try:
                        arg_i = schema_attr_names.index(attr)
                        if schema[arg_i]['domain'] == None:
                            schema[arg_i]['coded_values'] = uniques
                            schema[arg_i]['domain'] = 'coded_values_groupby'
                    except:
                        pass

                    attrs_stat[attr] = (stat, uniques)
                    if len(uniques) > max_len:
                        max_len = len(uniques)
                tar_df = pd.DataFrame()
                for attr in tar_schema:
                    uniques = attrs_stat[attr][1]
                    attrs_stat[attr] = attrs_stat[attr][0]
                    attr_vals = uniques + ['None'] * (max_len - len(uniques))
                    tar_df[attr] = attr_vals

                # collect stats first, also compare data types
                src_datatype = kb[concept]['matches'][datasource]['data_type']
                attr_schema = schema_set[datasource]
                cols_to_delete = bmm.compare_datatypes(src_datatype, attr_schema, tar_schema)
                tar_df = bmm.df_delete_cols(tar_df, cols_to_delete)

                # TODO datatypes must match! need to move to a different matcher
                sim_matrix, confidence = bmm.match_table_by_values(src_data, tar_df, params['match_threshold'], comparison_count_o, attrs_stat,
                                                                   params['sample_ratio'], params['sample_min_count'], params['sample_max_count'])
                print(sim_matrix.to_string())

                # save similarity matrices
                filename = '%s|%s|%s||%s.csv' % (concept, datasource, src_attr, source_name)
                sim_matrices[filename] = sim_matrix

        t3 = time.time()
        total = t3 - t2
        print('time %s sec' % (total))
        print('-----')

    return kb, sim_matrices

def update_kb(schema_set, kb, match_score_threshold):
    for root, dirs, files in os.walk(matching_output_p):
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
                print()
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
                    attr_vals, attr_datatype = rkc.get_example_values_from_schema(schema_set, tar_dataset, tar_attr)
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
                    rkc.add_attr_to_kb_concept(kb, matching_context)

            print()

    return kb

def find_new_concepts(schema_set, kb, datasources_with_tag):
    _, datasources, _, _ = find_datasources(datasources_with_tag)

    attr_schema = [schema_set[datasource] for datasource in datasources]

    attr_schema_parse = {datasource: [] for datasource in datasources}
    for datasource, schema in zip(datasources, attr_schema):
        for attr in schema:
            name = attr['name']
            attr_schema_parse[datasource].append(name)

    new_concepts = rkc.create_new_kb_concept(attr_schema_parse, kb, False)

    return new_concepts


if __name__ == "__main__":
    STAGES = [1,2,3,4,5]
    input_topics = ['trees', 'parks']
    input_datasets = []
    kb = None

    t0 = time.time()

    dataset_metadata_set, metadata_set, schema_set, datasources_with_tag = load_metadata(input_topics, input_datasets)
    datasources_index = datasources_with_tag

    if STAGES[0] == 1:
        print('-------INIT-------')
        kb, datasources_with_tag, datasources_index, reverse_index = find_datasources(datasources_with_tag)
        bmm.gather_statistics(schema_set, datasources_with_tag, dataset_stats, datasets_path)

        kb, datasources_with_tag, schema_set = initialize_matching( input_topics, dataset_metadata_set, schema_set,
                                                                    datasources_with_tag, reverse_index, kb)
        with open(schema_p, 'w') as fp:
            json.dump(schema_set, fp, sort_keys=True, indent=2)

        # kb_file = open(kb_file_p, "w")
        # json.dump(kb, kb_file, indent=2, sort_keys=True)


    if STAGES[1] == 2:
        # kb_f = open(kb_file_p, 'r')
        # kb = json.load(kb_f)

        params = {'match_threshold': 0.6,
                'sample_ratio': 0.01,
                'sample_min_count': 10,
                'sample_max_count': 100}

        print('-------MATCHING-------')
        kb, sim_matrices = perform_matching(dataset_metadata_set, schema_set, datasources_with_tag, kb, params)

        for filename, sim_matrix in sim_matrices.items():
            sim_matrix.to_csv(matching_output_p + filename, sep=',', encoding='utf-8')

        # kb_file = open(kb_file_p, "w")
        # json.dump(kb, kb_file, indent=2, sort_keys=True)


    if STAGES[2] == 3:
        # kb_f = open(kb_file_p, 'r')
        # kb = json.load(kb_f)

        match_score_threshold = 10

        print('-------UPDATE KB-------')
        kb = update_kb(schema_set, kb, match_score_threshold)

        # kb_file = open(kb_file_p, "w")
        # json.dump(kb, kb_file, indent=2, sort_keys=True)


    if STAGES[3] == 4:
        # kb_f = open(kb_file_p, 'r')
        # kb = json.load(kb_f)

        decision_threshold_split = 0.5
        decision_threshold_merge = 0.1

        print('-------SPLITTING-------')
        mappings_all = rkc.find_all_subtree_mappings(kb, False)
        # pprint.pprint(mappings_all)
        kb = rkc.split_concepts(kb, mappings_all, decision_threshold_split)

        print('-------MERGING-------')
        mappings_all = rkc.find_all_subtree_mappings(kb, False)
        # pprint.pprint(mappings_all)
        kb = rkc.merge_concepts(kb, mappings_all, decision_threshold_merge)

        kb_file = open(kb_file_p, "w")
        json.dump(kb, kb_file, indent=2, sort_keys=True)


    if STAGES[4] == 5:
        # kb_f = open(kb_file_p, 'r')
        # kb = json.load(kb_f)

        print(datasources_index)
        new_concepts = find_new_concepts(schema_set, kb, datasources_index)
        # pprint.pprint(new_concepts)

        new_concepts_f = open(new_concepts_p, "w")
        json.dump(new_concepts, new_concepts_f, indent=2, sort_keys=True)


    t1 = time.time()
    total = t1 - t0
    print('time %s sec' % (total))

    exit(0)
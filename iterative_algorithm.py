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

class Paths:
    datasets_path = './thesis_project_dataset_clean/'
    dataset_metadata_p = './inputs/datasource_and_tags.json'
    metadata_p = './inputs/metadata_tag_list_translated.json'
    schema_p = './inputs/schema_complete_list.json'
    matching_output_p = './outputs/instance_matching_output/'

    from time import gmtime, strftime
    curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    kb_file_p = "./outputs/kb_file_v1_"+curr_time+".json"

    kb_file_p_one_table_run = "./outputs/kb_file_v1_" + '{0}' + ".json"

    dataset_stats = './inputs/dataset_statistics/'
    new_concepts_p = "./outputs/new_concepts.json"
    new_concepts_f = './outputs/new_concepts.csv'


    # debug_datasources_with_tag = ['aquatic hubs', 'drainage 200 year flood plain', 'drainage water bodies','park specimen trees', 'parks', 'park screen trees']
    debug_datasources_with_tag = []

    weight_proportions = []

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

p = Paths()


def load_metadata(p, input_topics, input_datasets):
    '''TODO there might be a correct mapping between input_topics and attributes of input_datasets'''

    dataset_metadata_f = open(p.dataset_metadata_p, 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open(p.metadata_p, 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open(p.schema_p, 'r')
    schema_set = json.load(schema_f, strict=False)

    datasources_with_tag = {}
    for topic in input_topics:
        datasources_with_tag[topic]= metadata_set['tags'][topic]['sources']

    dataset_metadata_f.close()
    metadata_f.close()
    schema_f.close()

    return dataset_metadata_set, metadata_set, schema_set, datasources_with_tag


def find_datasources(p, datasources_with_tag, input_topics, kb):

    reverse_index = {}
    for topic in input_topics:
        for data_src in datasources_with_tag[topic]:
            if data_src not in p.debug_datasources_with_tag: continue

            path = p.datasets_path + data_src + '.csv'
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

def initialize_matching(p, input_topics, dataset_metadata_set, schema_set, datasources_with_tag, reverse_index, kb):

    datasources = {}
    for source_name in datasources_with_tag:
        # path = datasets_path + source_name + '.csv'
        # dataset = pd.read_csv(path, index_col=0, header=0)
        stats_f = open(p.dataset_stats + source_name + '.json', 'r')
        stats = json.load(stats_f)
        df_columns = list(stats.keys())

        schema = schema_set[source_name]
        metadata = dataset_metadata_set[source_name]['tags']
        dataset = pd.DataFrame()

        # dataset = bmm.df_rename_cols(dataset)

        datasources[source_name] = (source_name, dataset, schema, metadata)

        print(source_name)
        if DEBUG_MODE: bmm.print_metadata_head(source_name, dataset, schema, metadata)

        # initialization schema matching
        tags_list = [tag['display_name'] for tag in metadata]
        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
        cols_to_delete = bmm.find_attrs_to_delete(attributes_list, df_columns)
        attributes_list = [item for item in attributes_list if item not in cols_to_delete]

        sim_matrix = bmm.build_local_similarity_matrix(tags_list, attributes_list)
        sim_frame = pd.DataFrame(data=sim_matrix, columns=attributes_list, index=tags_list)

        # print(sim_frame.to_string())

        # TODO during new concepts stage, add second best tag and so on
        attrs = list(sim_frame.columns.values)

        # if stats file is empty
        if len(attrs) == 0:
            return kb, datasources_with_tag, schema_set

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
                if len(arg_max_examples_vals) > 0: example_value = arg_max_examples_vals[0]
            else:
                # loading from stats file
                _, uniques = bmm.get_attr_stats(p.dataset_stats, source_name, attrs[arg_i])
                # stat, _, uniques = bmm.groupby_unique(attrs[arg_i], dataset)

                uniques.sort()
                schema[arg_i]['coded_values'] = uniques
                arg_max_examples_vals = schema[arg_i]['coded_values']

                if len(arg_max_examples_vals) > 0: print('arg_max_examples_vals', arg_max_examples_vals[0])

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


import schema_matchers as schm
def perform_matching(p, dataset_metadata_set, schema_set, datasources_with_tag, kb, params):
    comparison_count = 0
    comparison_count_o = [comparison_count]
    sim_matrices = {}

    for source_name in datasources_with_tag:
        t2 = time.time()

        dataset = pd.read_csv(p.datasets_path + source_name + '.csv', index_col=0, header=0)
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
                    stat, uniques = bmm.get_attr_stats(p.dataset_stats, source_name, attr)
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

                src_names = list(src_data.columns.values)
                tar_names = list(tar_df.columns.values)
                sim_matrix2 = schm.matcher_name_matrix(src_names,tar_names)

                sim_matrix = schm.combine_scores_matrix(sim_matrix, sim_matrix2, params['proportions'])

                print(sim_matrix.to_string())

                # save similarity matrices
                filename = '%s|%s|%s||%s.csv' % (concept, datasource, src_attr, source_name)
                sim_matrices[filename] = sim_matrix

        t3 = time.time()
        total = t3 - t2
        print('time %s sec' % (total))
        print('-----')

    return kb, sim_matrices

def update_kb(p, schema_set, kb, match_score_threshold):
    for root, dirs, files in os.walk(p.matching_output_p):
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

def find_new_concepts(p, metadata_set, schema_set, kb, datasources_with_tag, num, input_topics):
    _, datasources, _, _ = find_datasources(p, datasources_with_tag, input_topics, kb)

    attr_schema = [schema_set[datasource] for datasource in datasources]

    attr_schema_parse = {datasource: [] for datasource in datasources}
    for datasource, schema in zip(datasources, attr_schema):
        for attr in schema:
            name = attr['name']
            attr_schema_parse[datasource].append(name)

    new_concepts = rkc.create_new_kb_concept(attr_schema_parse, kb, False)

    new_concepts, new_concepts_mod, concept_sims_scores, new_concepts_mod_df = rkc.select_new_concepts(metadata_set, schema_set, new_concepts, kb, num)

    return new_concepts, new_concepts_mod, concept_sims_scores, new_concepts_mod_df


def prepare_next_iteration(kb, output_new_concepts, p):
    breakout = False

    input_topics = output_new_concepts
    dataset_metadata_set, metadata_set, schema_set, datasources_with_tag = load_metadata(p, input_topics, None)

    # check if datasets are from the datasets of interest, if none then breakout
    count = 0
    for top in input_topics:
        datasources_top = datasources_with_tag[top]
        for ds in datasources_top:
            if ds in p.debug_datasources_with_tag:
                count += 1
    if count == 0:
        print('END OF ITERATIONS')
        breakout = True
    print('NUM OF NEW CONCEPTS:', len(input_topics))
    print('NUM OF TOPICS FOR DATASETS:', count)

    kb, datasources_with_tag, datasources_index, reverse_index = find_datasources(p, datasources_with_tag,
                                                                                     input_topics, kb)
    bmm.gather_statistics(schema_set, datasources_with_tag, p.dataset_stats, p.datasets_path)

    kb, datasources_with_tag, schema_set = initialize_matching(p, input_topics, dataset_metadata_set, schema_set,
                                                               datasources_with_tag, reverse_index, kb)

    return kb, schema_set, breakout


def one_full_run(input_topics):
    STAGES = [1,1,1,1,1,5]
    # STAGES = [1,0,0,0,1,1]

    # input_topics = ['trees', 'park']        # let's say we have 'park screen trees' to start with
    input_datasets = [] # not used
    kb = {}

    break_out = False

    dataset_metadata_set, metadata_set, schema_set, datasources_with_tag = load_metadata(p, input_topics, input_datasets)
    datasources_index = datasources_with_tag

    if STAGES[0] != 0:
        kb, datasources_with_tag, datasources_index, reverse_index = find_datasources(p, datasources_with_tag, input_topics, kb)
        bmm.gather_statistics(schema_set, datasources_with_tag, p.dataset_stats, p.datasets_path)

        print('-------INIT-------')
        kb, datasources_with_tag, schema_set = initialize_matching( p, input_topics, dataset_metadata_set, schema_set,
                                                                    datasources_with_tag, reverse_index, kb)
        # with open(p.schema_p, 'w') as fp:
        #     json.dump(schema_set, fp, sort_keys=True, indent=2)

    while True:

        # TODO check if topics are from the datasets of interest, if not then break
        if break_out: break
        print('===== ITERATIONS TO GO', STAGES[5], '=====')

        if STAGES[1] != 0:

            params = {'match_threshold': 0.6,
                    'sample_ratio': 0.01,
                    'sample_min_count': 10,
                    'sample_max_count': 100,
                    'proportions': p.weight_proportions  }

            print('-------MATCHING-------')
            kb, sim_matrices = perform_matching(p, dataset_metadata_set, schema_set, datasources_with_tag, kb, params)

            for filename, sim_matrix in sim_matrices.items():
                sim_matrix.to_csv(p.matching_output_p + filename, sep=',', encoding='utf-8')


        if STAGES[2] != 0:
            # kb_f = open(p.kb_file_p, 'r')
            # kb = json.load(kb_f)

            match_score_threshold = 10

            print('-------UPDATE KB-------')
            print(kb.keys(), len(kb))
            kb = update_kb(p, schema_set, kb, match_score_threshold)


        if STAGES[3] != 0:

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


        if STAGES[4] != 0:

            num_of_new_concepts = 5

            print('-------NEW CONCEPTS-------')
            new_concepts, new_concepts_mod, concept_sims_scores, df = find_new_concepts(p, metadata_set, schema_set, kb, datasources_index, num_of_new_concepts, input_topics)
            output_new_concepts = new_concepts_mod
            # print(datasources_index)
            # pprint.pprint(new_concepts)
            # print(new_concepts_mod)

            # df.to_csv(p.new_concepts_f, sep=',', encoding='utf-8')

            # new_concepts_f = open(p.new_concepts_p, "w")
            # json.dump(new_concepts, new_concepts_f, indent=2, sort_keys=True)

            kb, schema_set, break_out = prepare_next_iteration(kb, output_new_concepts, p)
            if break_out: break

            # with open(p.schema_p, 'w') as fp:
            #     json.dump(schema_set, fp, sort_keys=True, indent=2)

            # kb_file = open(p.kb_file_p, "w")
            # json.dump(kb, kb_file, indent=2, sort_keys=True)

        STAGES[5] = STAGES[5] - 1
        if STAGES[5] == 0: break


    kb_file = open(p.kb_file_p_one_table_run, "w")
    json.dump(kb, kb_file, indent=2, sort_keys=True)

    return

import pathlib
if __name__ == "__main__":
    table_topics_p = 'outputs/table_topics.json'
    table_setup_p = 'outputs/table_setup.json'

    f = open(table_topics_p)
    table_topics = json.load(f)

    f = open(table_setup_p)
    table_setup = json.load(f)

    p.weight_proportions = [0.8,0.2] # TODO change
    p.debug_datasources_with_tag = table_setup['tables']

    dataset_metadata_f = open('./inputs/datasource_and_tags.json', 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    t0 = time.time()

    table_setup['guiding_tables'] = {'parks': ['parks']} # TODO <<<=== this is the base table, change every time

    for k,table in enumerate(table_setup['guiding_tables']):
        if k % 2 != 0: continue # for now just try 5 tables

        dataset_name = table_setup['guiding_tables'][table][0]
        print(dataset_name)
        input_topics = [tag['display_name'] for tag in dataset_metadata_set[dataset_name]['tags']]

        # TODO change scope of datasets, per sample size per guiding table
        print('[[[',dataset_name,str(input_topics),']]]')
        for plan in table_topics[dataset_name]['samples']:
            if int(plan) > 10:
                continue
            mixes = table_topics[dataset_name]['samples'][plan]
            mixes = {'1+4':[['parks' ],['heritage sites', 'water utility facilities', 'sanitary lift stations', 'drainage dyke infrastructure']], '3+2':[['parks', 'park outdoor recreation facilities', 'park sports fields' ],['water assemblies', 'road row requirements downtown']]} # TODO <<<===
            for mix in mixes:
                p.debug_datasources_with_tag = mixes[mix][0] + mixes[mix][1]
                print('one_full_run:', plan, mix, p.debug_datasources_with_tag)

                json_kb_save_name = "./outputs/kb_file_v1_" + '{0}' + ".json"
                json_kb_save_name = json_kb_save_name.replace('{0}', dataset_name+'_'+plan+'_'+mix)
                p.kb_file_p_one_table_run = json_kb_save_name

                my_file = pathlib.Path(json_kb_save_name)
                if my_file.exists():
                    print('exists, skipping')
                    continue

                one_full_run(input_topics)

    t1 = time.time()
    total = t1 - t0
    print('TOTAL time %s sec' % (total))
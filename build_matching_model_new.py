import json
import pandas as pd

import parse_dataset as pds
import build_matching_model as bmm
import schema_matchers as sch
import preprocess_topic as pt

def load_metadata(p, m):
    '''TODO there might be a correct mapping between input_topics and attributes of input_datasets'''

    dataset_metadata_f = open(p.dataset_metadata_p, 'r')
    dataset_metadata_set = json.load(dataset_metadata_f)

    metadata_f = open(p.metadata_p, 'r')
    metadata_set = json.load(metadata_f)

    schema_f = open(p.schema_p, 'r')
    schema_set = json.load(schema_f, strict=False)

    dataset_metadata_f.close()
    metadata_f.close()
    schema_f.close()

    m.dataset_metadata_set = dataset_metadata_set
    m.metadata_set = metadata_set
    m.schema_set = schema_set

    return

from similarity.ngram import NGram
twogram = NGram(2)
fourgram = NGram(4)
import numpy as np

from similarity.metric_lcs import MetricLCS
metric_lcs = MetricLCS()
def build_local_similarity_matrix(source_schema, target_schema):
    source_schema_name = list(source_schema.keys())

    matrix= np.zeros((len(source_schema_name), len(target_schema)))

    for i in range(len(source_schema_name)):
            for j in range(len(target_schema)):
                # TODO call matcher
                sim_score = 1 - twogram.distance(source_schema_name[i],target_schema[j])
                # matrix[i,j] = np.int(100*SequenceMatcher(None,source_schema[i],target_schema[j]).ratio())
                matrix[i, j] = sim_score

                if matrix[i, j] >= 0.5:
                    print('matrix[i, j]', source_schema_name[i], target_schema[j], matrix[i, j])

                # if target_schema[j] == 'tree_species':
                #     print(source_schema[i], target_schema[j], matrix[i, j])

    return matrix

import random
def create_attributes_contexts(datasets, m, p, r):
    contexts = {}

    for dataset in datasets:
        contexts[dataset] = {}

        schema = m.schema_set[dataset]
        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]



        dataset_existing_tags = m.dataset_metadata_set[dataset]['tags']
        dataset_existing_groups = m.dataset_metadata_set[dataset]['groups']
        dataset_notes = m.dataset_metadata_set[dataset]['notes']

        desc = ''
        for group in dataset_existing_groups:
            desc = ' ' + group['description']

        dataset_existing_tags = [tag['display_name'] for tag in dataset_existing_tags]
        dataset_existing_groups = [group['display_name'] for group in dataset_existing_groups]
        dataset_notes = [word for word in dataset_notes.split() if "http://" not in word]

        notes = ' '.join(dataset_notes)

        stats_f = open(p.dataset_stats + dataset + '.json', 'r')
        stats = json.load(stats_f)

        df_columns = list(stats.keys())

        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
        cols_to_delete = bmm.find_attrs_to_delete(attributes_list, df_columns)
        attributes_list = [item for item in attributes_list if item not in cols_to_delete]


        for attr in attributes_list:
            # other_attrs = attributes_list.copy()
            # other_attrs.remove(attr)
            other_attrs = []

            attr_values = stats[attr].keys()
            # TODO get average of val length, place attr vals in notes if length is long

            length = 0
            if len(attr_values) > 0:


                if len(attr_values) > r.vals_truncate_sample:

                    num_to_select = r.vals_truncate_sample
                    attr_values = random.sample(attr_values, num_to_select)


                length = [len(val) for val in attr_values]
                length = sum(length)/len(attr_values)

            if r.sentence_threshold <= length:
                notes = notes + '. ' + '. '.join([val for val in attr_values])
                print('>>>>>', notes)
            else:
                other_attrs.extend(attr_values)
                print('>>>>>', other_attrs)

            pt.enrich_homonyms(dataset, attr, desc, notes, other_attrs)


    m.dataset_attributes_contexts = contexts
    return contexts

def topic_attribute_overlap():
    score = 0
    return score

def build_local_context_similarity_matrix(topics_contexts, attributes_contexts):
    topic_names = list(topics_contexts.keys())
    attribute_names = list(attributes_contexts.keys())

    matrix= np.zeros((len(topic_names), len(attribute_names)))

    for i in range(len(topic_names)):
            for j in range(len(attribute_names)):
                # call matchers

                sim_score = 0
                sim_score_arr = [0,0]
                sim_score_arr[0] = sch.matcher_name(topic_names[i],attribute_names[j], twogram)
                # sim_score = 1 - twogram.distance()

                # combine scores

                matrix[i, j] = sim_score

                # if matrix[i, j] >= 0.5:
                #     print('matrix[i, j]', topic_names[i], attribute_names[j], matrix[i, j])

    return matrix


def initialize_matching(p, m, r):

    datasources = {}
    for source_name in m.datasources_with_tag:
        # path = datasets_path + source_name + '.csv'
        # dataset = pd.read_csv(path, index_col=0, header=0)
        stats_f = open(p.dataset_stats + source_name + '.json', 'r')
        stats = json.load(stats_f)
        df_columns = list(stats.keys())

        schema = m.schema_set[source_name]
        metadata = m.dataset_metadata_set[source_name]['tags']
        dataset = pd.DataFrame()

        # dataset = bmm.df_rename_cols(dataset)

        datasources[source_name] = (source_name, dataset, schema, metadata)

        print(source_name)
        bmm.print_metadata_head(source_name, dataset, schema, metadata)

        # initialization schema matching
        tags_list = [tag['display_name'] for tag in metadata]
        # use enriched tags instead
        tags_list_enriched_f = open(p.enriched_topics_json_dir, 'r')
        tags_list_enriched = json.load(tags_list_enriched_f)
        tags_list_enriched_dataset = tags_list_enriched[source_name]
        tags_list_enriched_names = list(tags_list_enriched[source_name].keys())
        # TODO add homonyms to context



        attributes_list = [pds.clean_name(attr['name'], False, False) for attr in schema]
        cols_to_delete = bmm.find_attrs_to_delete(attributes_list, df_columns)
        attributes_list = [item for item in attributes_list if item not in cols_to_delete]



        sim_matrix = build_local_similarity_matrix(tags_list_enriched_dataset, attributes_list)
        # TODO build_local_similarity_matrix using context
        sim_matrix = build_local_context_similarity_matrix(topics_contexts, attributes_contexts)

        sim_frame = pd.DataFrame(data=sim_matrix, columns=attributes_list, index=tags_list_enriched_names)

        # print(sim_frame.to_string())

        attrs = list(sim_frame.columns.values)

        # if stats file is empty
        if len(attrs) == 0:
            print('empty dataset', source_name)
            continue


        # get example values
        for attr_i in range(len(schema)):
            print(attr_i)
            if schema[attr_i]['domain'] == None:

                attr_name = schema[attr_i]['name']
                attr_name = pds.clean_name(attr_name, False, False)

                # loading from stats file
                _, uniques = bmm.get_attr_stats(p.dataset_stats, source_name, attr_name)
                if uniques != None:
                    print('uniques', len(uniques))
                else:
                    continue

                # stat, _, uniques = bmm.groupby_unique(attrs[arg_i], dataset)

                uniques.sort()
                schema[attr_i]['coded_values'] = uniques
                # arg_max_examples_vals = schema[attr_i]['coded_values']

                # if len(arg_max_examples_vals) > 0: print('arg_max_examples_vals', arg_max_examples_vals[0])

                schema[attr_i]['domain'] = 'coded_values_groupby'

        # init kb
        build_kb_json(tags_list_enriched_names, source_name, m)

        # during new concepts stage, add second best tag and so on
        for topic in tags_list_enriched_names:
            scores = [[attr_i, attrs[attr_i], sim_frame.loc[topic, attrs[attr_i]]] for attr_i in range(len(attrs))]
            scores = sorted(scores, key=lambda tup: tup[2])
            scores.reverse()
            scores_examples = []
            for attr_score in scores:
                # example_value = None
                # print(attr_score, attr_score[0], schema[attr_score[0]])
                if 'coded_values' not in schema[attr_score[0]]:
                    continue
                arg_max_examples_vals = schema[attr_score[0]]['coded_values']
                arg_max_examples_vals.sort()
                scores_examples.append(attr_score + [schema[attr_score[0]]['coded_values']] )
                # print('here')

            top = 0
            output = []
            for score in scores_examples:
                if len(score) == 3:
                    print('skip', score)
                    continue
                print('topic_to_attr_count', score[2], top)
                if score[2] > r.topic_to_attr_threshold and top <= r.topic_to_attr_count:
                    print('topic_to_attr_count', r.topic_to_attr_count)
                    output.append(score)
                    top += 1
            if len(output) == 0:
                output.append(scores_examples[0])

            # max_score = 0
            # arg_max_score = None
            # arg_i = -1
            # for attr_i in range(len(attrs)):
            #     attr = attrs[attr_i]
            #     score = sim_frame.loc[topic, attr]
            #     if score > max_score:
            #         max_score = score
            #         arg_max_score = attr
            #         arg_i = attr_i



            # if len(arg_max_examples_vals) > 0: example_value = arg_max_examples_vals[0]
            # print('best match:', topic, arg_max_score, max_score, example_value)

            print('=====output', output)

            for match in output:
                kb_match_entry = {'concept': topic,
                                  'datasource': source_name,
                                  'attribute': match[1],
                                  'match_score': match[2],
                                  'example_values': match[3],
                                  'data_type': schema[match[0]]['data_type']}

                update_kb_json(m.kbs[source_name], kb_match_entry)
        print('-----')

    # done initialization

    return True

def build_kb_json(list_of_concepts, dataset_name, m):

    kb = {}
    for concept in list_of_concepts:
        concept_name = concept
        if concept_name not in kb:
            kb[concept_name] = {}
            # kb[concept_name]['datasources'] = datasources
            # kb[concept_name]['matches'] = {}
        else:
            kb_concept = kb[concept_name]
            # kb_concept['datasources'].extend(datasources)
        # TODO remove duplicates

    m.kbs[dataset_name] = kb
    return

def update_kb_json(kb, match_entry):
    concept = match_entry['concept']
    datasource = match_entry['datasource']
    attribute = match_entry['attribute']
    match_score = match_entry['match_score']
    example_values = match_entry['example_values']
    data_type = match_entry['data_type']

    kb_concept = kb[concept]
    # kb_concept_matches = kb_concept['matches']
    # kb_concept_matches[datasource] =
    kb_concept[attribute] = {'attribute': attribute, 'match_score' : match_score, 'example_values' : example_values, 'data_type' : data_type}
    return

class Paths:
    datasets_path = './thesis_project_dataset_clean/'
    dataset_stats = './inputs/dataset_statistics/'

    dataset_metadata_p = './inputs/datasource_and_tags.json'
    metadata_p = './inputs/metadata_tag_list_translated.json'
    schema_p = './inputs/schema_complete_list.json'

    matching_output_p = './outputs/instance_matching_output/'
    kb_file_p = "./outputs/kb_file.json"

    # new_concepts_p = "./outputs/new_concepts.json"
    # new_concepts_f = './outputs/new_concepts.csv'

    enriched_topics_dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/enriched_topics/'
    enriched_topics_json_dir = "./outputs/dataset_topics_enriched.json"

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

p = Paths()

class Metadata:
    dataset_metadata_set = None
    metadata_set = None
    schema_set = None
    datasources_with_tag = None
    kbs = {}

    dataset_topics_contexts = None
    dataset_attributes_contexts = None

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

m = Metadata()

class Parameters:
    topic_to_attr_threshold = 0.4
    topic_to_attr_count = 3

    sentence_threshold = 30
    vals_truncate_sample = 100

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

r = Parameters()

# GLAV mapping for each dataset
m.datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees','parks']
load_metadata(p, m)

## TODO call this to generate enriched attrs before running the rest
# print('create_attributes_contexts:')
# create_attributes_contexts(m.datasources_with_tag, m, p, r)
# exit(0)


initialize_matching(p, m, r)
with open(p.schema_p, 'w') as fp:
    json.dump(m.schema_set, fp, sort_keys=True, indent=2)

with open(p.kb_file_p, 'w') as fp:
    json.dump(m.kbs, fp, sort_keys=True, indent=2)
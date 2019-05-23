import json
import pandas as pd

import parse_dataset as pds
import build_matching_model as bmm

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

def initialize_matching(p, m):

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
        tags_list = [tag['display_name'] for tag in metadata] # TODO
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
            return False

        for topic in tags_list:

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

            update_kb_json(kb, kb_match_entry)
        print('-----')

    # done initialization

    return True

def build_kb_json(list_of_concepts, dataset_name, m):
    m.kbs[dataset_name] = {}

    for concept in list_of_concepts:
        concept_name = concept[0]
        datasources = concept[1]
        if concept_name not in kb:
            kb[concept_name] = {}
            kb[concept_name]['datasources'] = datasources
            kb[concept_name]['matches'] = {}
        else:
            kb_concept = kb[concept_name]
            kb_concept['datasources'].extend(datasources)
        # TODO remove duplicates
    return kb

def update_kb_json(kb, match_entry):
    concept = match_entry['concept']
    datasource = match_entry['datasource']
    attribute = match_entry['attribute']
    match_score = match_entry['match_score']
    example_values = match_entry['example_values']
    data_type = match_entry['data_type']

    kb_concept = kb[concept]
    kb_concept_matches = kb_concept['matches']
    kb_concept_matches[datasource] = {'attribute': attribute, 'match_score' : match_score, 'example_values' : example_values, 'data_type' : data_type}
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

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

p = Paths()

class Metadata:
    dataset_metadata_set = None
    metadata_set = None
    schema_set = None
    datasources_with_tag = None
    kbs = {}

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

m = Metadata()

# GLAV mapping for each dataset
m.datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees','parks']
load_metadata(p, m)


kb, datasources_with_tag, schema_set = initialize_matching(p, m)
with open(p.schema_p, 'w') as fp:
    json.dump(schema_set, fp, sort_keys=True, indent=2)
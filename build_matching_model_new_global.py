import json
import pandas as pd

import build_matching_model_new as bmmn
import parse_dataset as pds
import build_matching_model as bmm
import preprocess_topic as pt

class Metadata2:
    guiding_table_name = None

    kbs = {}
    pair_dict_all = None

    enriched_attrs = None
    enriched_topics = None

    attrs_contexts = None
    topic_contexts = None
    all_topics = None

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

m2 = Metadata2()

class TableMetadata:
    table_name = None

    tags_list_enriched_dataset = None
    tags_list_enriched_names = None

    attributes_list = None
    schema = None
    attribute_contexts = None

    exposed_topics = None
    dataset_stats = None

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

bmmn.m.datasources_with_tag = ['aquatic hubs','drainage 200 year flood plain','drainage water bodies','park specimen trees', 'parks'] #

bmmn.load_metadata(bmmn.p, bmmn.m)

m2.all_topics, m2.attrs_contexts, m2.topic_contexts = bmmn.load_prematching_metadata(bmmn.p, bmmn.m, pds)

kb_file_f = open(bmmn.p.kb_file_const_p, 'r')
m2.kbs = json.load(kb_file_f)
kb_file_f.close()

enriched_attrs_f = open(bmmn.p.enriched_attrs_json_dir, 'r')
m2.enriched_attrs = json.load(enriched_attrs_f)
enriched_attrs_f.close()

enriched_topics_f = open(bmmn.p.enriched_topics_json_dir, 'r')
m2.enriched_topics = json.load(enriched_topics_f)
enriched_topics_f.close()

wordnet = pt.load_dict()

# load guiding table
m2.guiding_table_name = 'parks'

gtm = TableMetadata(table_name=m2.guiding_table_name)
gtm.tags_list_enriched_dataset, gtm.tags_list_enriched_names, gtm.attributes_list, gtm.schema, _, _ = bmmn.load_per_source_metadata(bmmn.p, bmmn.m, {}, m2.guiding_table_name, pds, bmm)
gtm.attribute_contexts = m2.attrs_contexts[m2.guiding_table_name]
gtm.exposed_topics = gtm.tags_list_enriched_names
gtm.dataset_stats = bmm.get_table_stats(bmmn.p.dataset_stats, m2.guiding_table_name)

# load all other tables


# compare guiding table topics with other tables topics and get most similar table
table_metadata = {}
table_scores = {}
for source in bmmn.m.datasources_with_tag:
    if source == m2.guiding_table_name: continue
    if source not in table_metadata:
        tm = TableMetadata(table_name=source)
        tm.tags_list_enriched_dataset, tm.tags_list_enriched_names, tm.attributes_list, tm.schema, _, _ = bmmn.load_per_source_metadata(
            bmmn.p, bmmn.m, {}, source, pds, bmm)

        if source not in m2.attrs_contexts: continue
        tm.attribute_contexts = m2.attrs_contexts[source]
        table_metadata[source] = tm

        tm.exposed_topics = tm.tags_list_enriched_names
        # TODO update tags_list_enriched_dataset when new attrs are added to exposed_topics. Remove attrs from exposed_topics at end of iter

        tm.dataset_stats = bmm.get_table_stats(bmmn.p.dataset_stats, source)

    table_scores[source] = [0, None]

    # dataset = pd.read_csv(bmmn.p.datasets_path + source + '.csv', index_col=0, header=0)
    # dataset = bmm.df_rename_cols(dataset)

    sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({source: gtm.tags_list_enriched_dataset},table_metadata[source].tags_list_enriched_dataset, source, wordnet, {})

    sim_matrix1 = bmmn.build_local_similarity_matrix(gtm.tags_list_enriched_dataset, table_metadata[source].exposed_topics, bmmn.r)

    # print(table_metadata[source].exposed_topics, gtm.exposed_topics)
    # print(gtm.tags_list_enriched_dataset.keys(), table_metadata[source].exposed_topics)

    # print(sim_matrix1.shape, sim_matrix2.shape, sim_matrix3.shape)

    sim_frame1 = pd.DataFrame(data=sim_matrix1, columns=table_metadata[source].exposed_topics, index=gtm.exposed_topics)
    sim_frame2 = pd.DataFrame(data=sim_matrix2, columns=table_metadata[source].exposed_topics, index=gtm.exposed_topics)
    sim_frame3 = pd.DataFrame(data=sim_matrix3, columns=table_metadata[source].exposed_topics, index=gtm.exposed_topics)

    # print(sim_frame2.head())

    frames = [sim_frame1, sim_frame2, sim_frame3]
    # TODO sim_frame3 doesn't look right

    cols = list(sim_frame1.columns.values)

    curr_max = 0
    arg_cur_max = []
    for i in range(len(frames)):
        frame = frames[i]
        for col in cols:
            max_row = frame[col].idxmax()
            max_val = sim_frame1.loc[max_row, col]

            # if max_val > 0: print([i, max_row, col, max_val])

            if max_val>curr_max:
                arg_cur_max = [i, max_row, col]
                curr_max = max_val

    print([curr_max, arg_cur_max])
    if curr_max > bmmn.r.table_similarity_thresh:
        table_scores[source] = [curr_max, arg_cur_max]

import pprint
pprint.pprint(table_scores)

# add new similar topics, transfer guiding table attrs to new topic group, and transfer other table attrs to existing guiding table topic group. use topic-topic wordnet, ngram name
comparing_pairs = {}
added_topics = {}
for dataset in table_scores:
    score = table_scores[dataset][0]
    info = table_scores[dataset][1]
    if info[1] == info[2]: continue # TODO change. need update all topics, not the best one
    gtm_kb = m2.kbs[m2.guiding_table_name]
    dataset_kb = m2.kbs[dataset]

    comparing_pairs[(info[1], dataset, info[2])] = (gtm_kb[info[1]].copy(), dataset_kb[info[2]].copy())

    # transfer
    update = dataset_kb[info[2]].copy()
    for attr in update:
        if 'source_dataset' not in dataset_kb[info[2]][attr]:
            dataset_kb[info[2]][attr]['source_dataset'] = dataset
    gtm_kb[info[1]].update(update)
    gtm_kb[info[2]] = gtm_kb[info[1]]


    pprint.pprint(gtm_kb[info[1]].keys())
    pprint.pprint(gtm_kb[info[2]].keys())

    if dataset not in added_topics:
        added_topics[dataset] = []
    added_topics[dataset].append(info[2])


# TODO check to see if any topics in other table can be mapped to non-assigned attrs in guiding table. use attr-topic wordnet, ngram name
# TODO check to see if any non-assigned attrs in other table can be mapped to guiding table topics. use attr-topic wordnet, ngram name
# compute attr-attr similarity matrix (just append one more column). use attr-attr comparison: tf-idf pair of document of values, TODO ngram per val, attr name wordnet
def preprocesss_attr_values(values):
    splits = []


    for value in values:
        value = str(value) # TODO get alpha to numeric ratio, if all numbers then skip
        value.replace('-', '')
        value.replace('.', '')
        val_spt = pt.splitter.split(value.lower())
        val_spt = [val for val in val_spt if 'http' not in val]
        # print(val_spt)
        val_spt_merge = []
        for item in val_spt:
            val_spt_merge.extend(item)
        splits.append(' '.join(val_spt_merge))

    return ' '.join(splits)
# preprocesss_attr_values(['I am splitting this text.','This some nonsense text qwertyuiop'])

import schema_matchers as sch

sim_dict = {}

for pair in comparing_pairs:
    gtm_top_name, dataset,dataset_top_name  = pair
    gtm_top, dataset_top = comparing_pairs[pair]

    for gtm_attr in gtm_top:
        for dataset_attr in dataset_top:

            text1 = preprocesss_attr_values(gtm_top[gtm_attr]['example_values'])
            text2 = preprocesss_attr_values(dataset_top[dataset_attr]['example_values'])

            if len(text1) == 0 or len(text2) == 0: continue

            score = sch.matcher_instance_document(text1, text2)

            if score > 0:
                print('>>>>>', gtm_attr, ' | ', dataset, dataset_attr)
                print(len(text1), len(text2), score)


# TODO clustering. split out attrs from topics (break ties usig avg attr-topic score), merge attrs into (newly added only) groups if possible

# remove added topics from the other table
pprint.pprint(added_topics)
for dataset in added_topics:
    rm_topics = added_topics[dataset]
    exposed_topics = table_metadata[dataset].exposed_topics
    for top in rm_topics:
        exposed_topics.remove(top)

    pprint.pprint(table_metadata[dataset].exposed_topics)

# TODO repeat comparing table topics

# transfer topics to tables related to guiding table using topic-attr mappings
import classification_evaluation as ce
dataset_topics = ce.kb_to_topics_per_dataset(m2.kbs[m2.guiding_table_name], m2.guiding_table_name)

print('=====transfer topics=====')
pprint.pprint(dataset_topics)

# eval accuracy
ground = {'parks' : ['green', 'trees', 'parks'], 'park specimen trees' : ['green', 'trees']}
accu = ce.compute_precision_and_recall(ground, [dataset_topics])

print('=====accuracy=====')
print(accu)
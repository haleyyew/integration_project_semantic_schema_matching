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
    exposed_topics_groups = {}

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

# TODO add enriched topics to dataset if the topic is in vocab
# TODO find more topics for nonmapped attrs

# load all other tables


# compare guiding table topics with other tables topics and get most similar table
def process_scores(sim_matrix1, sim_matrix2, sim_matrix3, table_metadata_topics, source, table_scores, gtm_topics):
    sim_frame1 = pd.DataFrame(data=sim_matrix1, columns=table_metadata_topics, index=gtm_topics)
    sim_frame2 = pd.DataFrame(data=sim_matrix2, columns=table_metadata_topics, index=gtm_topics)
    sim_frame3 = pd.DataFrame(data=sim_matrix3, columns=table_metadata_topics, index=gtm_topics)

    # print(sim_frame2.head())

    frames = [sim_frame1, sim_frame2, sim_frame3]
    # TODO sim_frame3 doesn't look right

    cols = list(sim_frame1.columns.values)

    list_of_scores = []

    curr_max = 0
    arg_cur_max = []
    for i in range(len(frames)):
        frame = frames[i]
        for col in cols:
            max_row = frame[col].idxmax()
            max_val = frame.loc[max_row, col]

            # if max_val > 0: print([i, max_row, col, max_val])
            list_of_scores.append([max_val, [i, max_row, col]])

            if max_val > curr_max:
                arg_cur_max = [i, max_row, col]
                curr_max = max_val

    #  NOTE: need update all topics, not the best one
    print(source, ' : ', [curr_max, arg_cur_max])

    if curr_max > bmmn.r.table_similarity_thresh:
        if source not in table_scores:
            table_scores[source] = []
        table_scores[source].extend(list_of_scores)

    return table_scores, list_of_scores


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

    table_scores[source] = [[0, None]]

    # dataset = pd.read_csv(bmmn.p.datasets_path + source + '.csv', index_col=0, header=0)
    # dataset = bmm.df_rename_cols(dataset)

    sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({source: gtm.tags_list_enriched_dataset},table_metadata[source].tags_list_enriched_dataset, source, wordnet, {})

    sim_matrix1 = bmmn.build_local_similarity_matrix(gtm.tags_list_enriched_dataset, table_metadata[source].exposed_topics, bmmn.r)

    # print(table_metadata[source].exposed_topics, gtm.exposed_topics)
    # print(gtm.tags_list_enriched_dataset.keys(), table_metadata[source].exposed_topics)

    # print(sim_matrix1.shape, sim_matrix2.shape, sim_matrix3.shape)

    table_scores, list_of_scores = process_scores(sim_matrix1, sim_matrix2, sim_matrix3, table_metadata[source].exposed_topics, source, table_scores, gtm.exposed_topics)



import pprint
print('===== table search =====')
pprint.pprint(table_scores)
print('==========')

def topic_topic_update(dataset, info, score, comparing_pairs, gtm, gtm_kb, dataset_kb, added_topics):
    # for fine grained matching later
    comparing_pairs[(info[1], dataset, info[2])] = (gtm_kb[info[1]].copy(), dataset_kb[info[2]].copy())

    # transfer
    update = dataset_kb[info[2]].copy()
    for attr in update:
        if 'source_dataset' not in dataset_kb[info[2]][attr]:
            update[attr]['source_dataset'] = [dataset]
        else:
            update[attr]['source_dataset'].append(dataset)

    gtm_kb[info[1]].update(update)
    gtm_kb[info[2]] = gtm_kb[info[1]]  # NOTE: put in one group instead

    # put dataset's topic into the same group as guiding table
    if info[1] not in gtm.exposed_topics_groups:
        gtm.exposed_topics_groups[info[1]] = []
    gtm.exposed_topics_groups[info[1]].append((dataset, info[2], score))

    print('----- transfer -----')
    pprint.pprint(gtm_kb[info[1]].keys())
    # pprint.pprint(gtm_kb[info[2]].keys())
    print('----------')

    if dataset not in added_topics:
        added_topics[dataset] = []
    # print('add', dataset, info[2])

    if info[2] not in added_topics[dataset]:
        added_topics[dataset].append(info[2])  # these topics will be removed later

    return comparing_pairs, gtm, gtm_kb, added_topics

# add new similar topics, transfer guiding table attrs to new topic group, and transfer other table attrs to existing guiding table topic group. use topic-topic wordnet, ngram name
comparing_pairs = {}
added_topics = {}
for dataset in table_scores:
    scores = sorted(table_scores[dataset], key=lambda x: x[0])
    scores.reverse()

    # print(':::::', dataset, scores)

    gtm_kb = m2.kbs[m2.guiding_table_name]
    dataset_kb = m2.kbs[dataset]

    for item in scores:
        if item[0] < bmmn.r.topic_to_attr_threshold:
            break

        score = item[0]
        info = item[1]
        if info[1] == info[2]:  # dataset and guiding have the same topic
            #  NOTE: still need to add it
            for attr in gtm_kb[info[1]]:
                if 'source_dataset' not in gtm_kb[info[1]][attr]:
                    gtm_kb[info[1]][attr]['source_dataset'] = [m2.guiding_table_name]
            pass

        # comparing_pairs, gtm, gtm_kb, added_topics = \
        topic_topic_update(dataset, info, score, comparing_pairs, gtm, gtm_kb, dataset_kb, added_topics)


print('----- end of transfer -----')
pprint.pprint(gtm.exposed_topics_groups)
print('----------')

for key in gtm.exposed_topics_groups:
    for tupl in gtm.exposed_topics_groups[key]:
        topic = tupl[1]
        if topic not in gtm.exposed_topics:
            gtm.exposed_topics.append(topic)

# pprint.pprint(gtm.exposed_topics)

# TODO check to see if any topics in other table can be mapped to non-assigned attrs in guiding table. use attr-topic wordnet, ngram name
def gt_nonmapped_attr_to_new_topic(table_scores, table_metadata, guiding_table):
    gt_attrs = guiding_table.attributes_list
    gt_exposed_topics = guiding_table.exposed_topics

    gt_attrs_mapped = []
    gtm_kb = m2.kbs[m2.guiding_table_name]


    for topic in gtm_kb:
        for attr in gtm_kb[topic]:

            if 'source_dataset' in gtm_kb[topic][attr] and m2.guiding_table_name in gtm_kb[topic][attr]['source_dataset']:
                gt_attrs_mapped.append(attr)
            else:
                gt_attrs_mapped.append(attr)

    pprint.pprint(gt_attrs_mapped)

    candidate_attrs = list(set(gtm.attribute_contexts.keys()) - set(gt_attrs_mapped))

    attribute_contexts_temp = guiding_table.attribute_contexts.copy()
    attribute_contexts_temp_keys = list(attribute_contexts_temp.keys())
    for attr in attribute_contexts_temp_keys:
        if attr not in candidate_attrs:
            del attribute_contexts_temp[attr]

    # print(attribute_contexts_temp.keys())

    table_scores_attr_topic = {}


    for dataset in table_scores:
        scores = sorted(table_scores[dataset], key=lambda x: x[0])
        scores.reverse()

        candidates = []
        for item in scores:
            # if item[0] < bmmn.r.topic_to_attr_threshold:
            #     break
            if item[0] == 0:
                break
            candidates.append(item[1][2])

        not_added = list(set(candidates) - set(gt_exposed_topics))

        curr_exposed_topics = list(table_metadata[dataset].tags_list_enriched_dataset.keys())

        # print(dataset, curr_exposed_topics, not_added, candidates, gt_exposed_topics)

        tags_list_enriched_temp = table_metadata[dataset].tags_list_enriched_dataset.copy()
        for topic in curr_exposed_topics:
            if topic not in not_added:
                del tags_list_enriched_temp[topic]

        table_scores_attr_topic[source] = [[0, None]]

        if len(tags_list_enriched_temp.keys()) == 0: continue
        # print(dataset, tags_list_enriched_temp.keys())


        sim_matrix2, sim_matrix3, _ = bmmn.build_local_context_similarity_matrix({dataset: tags_list_enriched_temp}, attribute_contexts_temp, dataset, wordnet, {})

        sim_matrix1 = bmmn.build_local_similarity_matrix(tags_list_enriched_temp, list(attribute_contexts_temp.keys()), bmmn.r)



        table_scores_attr_topic, list_of_scores = process_scores(sim_matrix1, sim_matrix2, sim_matrix3, list(attribute_contexts_temp.keys()), dataset,
                                                      table_scores_attr_topic, list(tags_list_enriched_temp.keys()))

    return table_scores_attr_topic

table_scores_attr_topic = gt_nonmapped_attr_to_new_topic(table_scores, table_metadata, gtm)

# print('///// topics attributes /////')
# pprint.pprint(table_scores_attr_topic)
# print('//////////')

for dataset in table_scores_attr_topic:
    if dataset == m2.guiding_table_name: continue

    scores = sorted(table_scores_attr_topic[dataset], key=lambda x: x[0])
    scores.reverse()

    # print(':::::', dataset, scores)

    gtm_kb = m2.kbs[m2.guiding_table_name]
    dataset_kb = m2.kbs[dataset]
    # print(dataset_kb.keys())

    for item in scores:
        if item[0] < bmmn.r.topic_to_attr_threshold:
            break

        score = item[0]
        info = item[1]

        if info[1] not in gtm_kb:
            gtm_kb[info[1]] = {}

        attr = info[2]
        # print('/////', dataset, info, score)
        gtm_kb[info[1]][attr] = {}

        dataset_schema = gtm.schema
        # print(len(dataset_schema))
        index = -1
        search = info[2].replace(' ', '_')
        datatype = None
        examples = None
        for sch_attr in dataset_schema:
            if sch_attr['name'] == search or sch_attr['alias'] == info[2]:

                if 'coded_values' in sch_attr:
                    examples = sch_attr['coded_values']

                datatype = sch_attr['data_type']

        kb_match_entry = {'concept': info[1],
                          'datasource': dataset,
                          'attribute': info[2],
                          'match_score': score,
                          'example_values': examples,
                          'data_type': datatype,
                          'score_name': bmmn.m.score_names[info[0]]}

        bmmn.update_kb_json(gtm_kb, kb_match_entry)

        # kb_match_entry['example_values'] = kb_match_entry['example_values'][
        #                                    :min(len(kb_match_entry['example_values']), 5)]
        # pprint.pprint(kb_match_entry)

        # gtm_kb[info[1]][attr] = dataset_kb[info[1]][attr]
        #
        # update = gtm_kb[info[1]]
        # if 'source_dataset' not in update[attr]['source_dataset']:
        #     update[attr]['source_dataset'] = [dataset]
        # else:
        #     update[attr]['source_dataset'].append(dataset)

        if info[1] not in added_topics[dataset]:
            added_topics[dataset].append(info[1])

print('----- end of coverage -----')
pprint.pprint(added_topics)
print('----------')


# TODO check to see if any non-assigned attrs in other table can be mapped to guiding table topics. use attr-topic wordnet, ngram name
def gt_topic_to_dataset_nonmapped_attr():
    return

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

# fine grained matching
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
                print('>>>>>', gtm_attr, len(text1) ,' | ', dataset, dataset_attr, len(text2),  ' || ',  score)
                print('<<<<<')


# TODO clustering. split out attrs from topics (break ties usig avg attr-topic score), merge attrs into (newly added only) groups if possible

# remove added topics from the other table
print('=====remove added topics=====')
pprint.pprint(added_topics)
for dataset in added_topics:
    rm_topics = added_topics[dataset]
    exposed_topics = table_metadata[dataset].exposed_topics
    for top in rm_topics:
        # print('rm', dataset, top)
        exposed_topics.remove(top)

    pprint.pprint(table_metadata[dataset].exposed_topics)

# TODO repeat comparing table topics

# transfer topics to tables related to guiding table using topic-attr mappings
import classification_evaluation as ce
dataset_topics = ce.kb_to_topics_per_dataset(m2.kbs[m2.guiding_table_name], m2.guiding_table_name)

# add more topics that have no attrs mapped, NOTE: every dataset in group gets all topics in group
def add_more_topics(exposed_topics_groups, dataset_topics):
    for key in exposed_topics_groups:
        items_list = exposed_topics_groups[key]

        unique_topics = []
        unique_datasets = []
        for tupl in items_list:
            dataset, topic, score = tupl[0], tupl[1], tupl[2]
            if topic not in unique_topics:
                unique_topics.append(topic)

            if dataset not in unique_datasets:
                unique_datasets.append(dataset)

        # for tupl in items_list:
        #     dataset, topic, score = tupl[0], tupl[1], tupl[2]
        for dataset in unique_datasets:
            if dataset not in dataset_topics:
                dataset_topics[dataset] = []
            if m2.guiding_table_name not in dataset_topics:
                dataset_topics[m2.guiding_table_name] = []

            for topic in unique_topics:
                if topic not in dataset_topics[dataset]:
                    dataset_topics[dataset].append(topic)
                if topic not in dataset_topics[m2.guiding_table_name]:
                    dataset_topics[m2.guiding_table_name].append(topic)
    return

# TODO remember to remove topics from here if found some topic should be split
add_more_topics(gtm.exposed_topics_groups, dataset_topics)

print('=====transfer topics=====')
pprint.pprint(dataset_topics)

# eval accuracy
ground = {'parks' : ['green', 'trees', 'parks'], 'park specimen trees' : ['green', 'trees']}
accu = ce.compute_precision_and_recall(ground, [dataset_topics])

print('=====accuracy=====')
print(accu)
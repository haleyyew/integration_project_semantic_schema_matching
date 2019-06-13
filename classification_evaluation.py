labels = ['l1', 'l2', 'l3', 'l4', 'l5']
tbl1_attrs = ['t1a2', 't1a3', 't1a4', 't1a6']
tbl2_attrs = ['t2a1', 't2a2', 't2a3', 't2a4', 't2a6']
tbl3_attrs = ['t3a1', 't3a3']

sem_dist = {}
sem_dist[('l1', 'l5')] = 0.5
sem_dist[('l2', 'l5')] = 0.6
sem_dist[('l3', 'l5')] = 0.7
sem_dist[('l4', 'l5')] = 0.8
# etc

ground = {'l1': ['t2a1', 't3a1'], 'l2': ['t1a2', 't2a2'], 'l3': ['t1a3', 't2a3', 't3a3'], 'l4': ['t1a4', 't2a4']}
top1 = {'l1': ['t2a6', 't3a1'], 'l2': ['t1a2', 't2a2'], 'l3': ['t1a3', 't2a3'], 'l5': ['t1a6', 't2a4']}
top2 = {'l2': ['t2a6'], 'l3': ['t1a3'], 'l5': ['t1a4', 't2a4']}

def kb_to_topics_per_dataset(kb, guiding_name):
    datasets = {guiding_name: []}

    # print(kb.keys())

    for topic in kb:
        # print(kb[topic].keys())
        for attr in kb[topic]:


            if 'source_dataset' in kb[topic][attr]:
                datasets_list = kb[topic][attr]['source_dataset']   # list of datasets []

                # print(topic, attr, datasets_list)
                for dataset in datasets_list:
                    if dataset not in datasets: datasets[dataset] = []

                    datasets[dataset].append(topic)
            else:
                datasets[guiding_name].append(topic)

    for dataset in datasets:
        datasets[dataset] = list(set(datasets[dataset]))

    return datasets

def compute_precision_and_recall(ground, set_of_labeling_set):
    # print(ground)
    # print(set_of_labeling_set)

    all_labels = []
    for labeling in set_of_labeling_set:
        keys = labeling.keys()
        all_labels.extend(list(keys))
    all_labels = list(set(all_labels))

    ground_truth_reverse_index = {}
    for label in ground:
        for attr in ground[label]:
            if attr not in ground_truth_reverse_index: ground_truth_reverse_index[attr] = []
            if label not in ground_truth_reverse_index[attr]: ground_truth_reverse_index[attr].append(label)

    num_ground_truth_labelings = len([label for label in ground for attr in ground[label]])

    max_labeling_len = 0    # for precision
    for labeling in set_of_labeling_set:
        labeling_len = sum([1 for label in labeling for attr in labeling[label]])
        if labeling_len > max_labeling_len: max_labeling_len = labeling_len

    correct = {}
    for label in all_labels:
        for labeling in set_of_labeling_set:
            if label in labeling:
                attrs = labeling[label]
                for attr in attrs:
                    if attr in ground_truth_reverse_index:
                        if label in ground_truth_reverse_index[attr]:   # TODO check semantic distance too
                            correct[(attr, label)] = label   # TODO value is score for semantic distance

    # print('num_ground_truth_labelings', num_ground_truth_labelings)
    # print(correct)
    # precision and recall
    return len(correct.keys())/max_labeling_len, len(correct.keys())/num_ground_truth_labelings


def v1_transfer_topics(kb):
    kb_topics = {}

    for topic in kb:
        topic_matches = kb[topic]['matches']

        if topic_matches == None: continue
        if len(topic_matches) == 0: continue

        if topic not in kb_topics: kb_topics[topic] = []
        for ds in topic_matches:
            kb_topics[topic].append(ds)

    kb_topics_reverse = {}
    for topic in kb_topics:
        for ds in kb_topics[topic]:
            if ds not in kb_topics_reverse: kb_topics_reverse[ds] = []
            kb_topics_reverse[ds].append(topic)

    return kb_topics, kb_topics_reverse

import os
import pickle
def get_ground(dir_p):
    # print(dir_p)
    ground = {}
    for root, dirs, files in os.walk(dir_p):
        # print(root, dirs, files)
        for file in files:
            filename, file_extension = os.path.splitext(file)

            # print(filename)
            filename_split = filename.split('_')[-1]
            filename_split = filename_split.replace('[', '')
            filename_split = filename_split.replace(']', '')
            # print(filename_split)

            with open(dir_p + file, 'rb') as fp:
                topics_new = pickle.load(fp)

            ground[filename_split] = topics_new

    return ground

import build_matching_model_new_global as bmmng
import json
if __name__ == "__main__":
    precision, recall = compute_precision_and_recall(ground, [top1, top2])
    # print(precision, recall)

    # eval for ['aquatic hubs', 'drainage 200 year flood plain', 'drainage water bodies',
    #                               'park specimen trees', 'parks', 'park screen trees']
    # TODO make this set more complete
    # ground = {'parks' : ['green', 'trees', 'parks'], 'park specimen trees' : ['green', 'trees']}
    existing_tags_p = '/Users/haoran/Documents/thesis_schema_integration/inputs/datasource_and_tags.json'
    updated_topics_p = '/Users/haoran/Documents/thesis_schema_integration/outputs/updated_topics/'
    # print('here')
    ground = get_ground(updated_topics_p)
    # ground['park screen trees'] = []    # TODO
    # print(ground)
    # exit(0)

    metadata_f = open(existing_tags_p, 'r')
    metadata = json.load(metadata_f)

    existing_tags = {}
    for ds in ground.keys():
        if ds not in existing_tags: existing_tags[ds] = []
        for tag in metadata[ds]['tags']:
            # print(tag)
            name = tag['display_name']
            existing_tags[ds].append(name)
    # ground['park screen trees'] = existing_tags['park screen trees']    # TODO

    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    # version 1

    kb_p = dir+'kb_file_v1.json'
    kb_topics_f = open(kb_p, 'r')
    kb_topics_json = json.load(kb_topics_f)

    kb_topics, kb_topics_reverse = v1_transfer_topics(kb_topics_json)
    # print(kb_topics)
    accu = compute_precision_and_recall(bmmng.reverse_dict(ground), [kb_topics])
    print('ver1', accu)

    num_new = 0
    ds_count = 0
    for ds in kb_topics_reverse:
        ds_count += 1
        num_new += len(set(kb_topics_reverse[ds]) - set(existing_tags[ds]))
        print(ds, set(kb_topics_reverse[ds]) - set(existing_tags[ds]), set(kb_topics_reverse[ds]) - set(existing_tags[ds]))
    print('avg new', num_new/ds_count)

    # version 2


    dataset_topics_p = dir+'dataset_topics_v2.json'
    dataset_topics_f = open(dataset_topics_p, 'r')
    dataset_topics = json.load(dataset_topics_f)
    # print(dataset_topics)

    accu = compute_precision_and_recall(bmmng.reverse_dict(ground), [bmmng.reverse_dict(dataset_topics)])
    print('ver2', accu)

    num_new = 0
    ds_count = 0
    for ds in dataset_topics:
        ds_count += 1
        num_new += len(set(dataset_topics[ds]) - set(existing_tags[ds]))
        print(ds, set(dataset_topics[ds]) - set(existing_tags[ds]), set(existing_tags[ds]) - set(dataset_topics[ds]))
    print('avg new', num_new/ds_count)

    # TODO also calculate num of new topics created per dataset
    # plan out how to evaluate

#-----


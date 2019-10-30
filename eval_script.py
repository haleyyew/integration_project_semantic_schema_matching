from classification_evaluation import compute_precision_and_recall
from build_matching_model_new_global import reverse_dict

import pickle
import json
import pprint
import os.path

ground_truth_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/updated_topics/'
eval_inputs_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/eval_inputs.json'
eval_true_tables_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/eval_true_tables.json'

brute_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/brute_force_json/'
iter_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/iter/'
data_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/data_driven_output/'
iter_data_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/iter+data/'
iter_data_noch_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/iter+data_no_check/'

guiding_table = 'parks'
print('guiding table(s):', guiding_table)

with open(eval_inputs_path, 'r') as eval_inputs_f:
    eval_inputs = json.load(eval_inputs_f)

with open(eval_true_tables_path, 'r') as eval_true_tables_f:
    truth_tables_set = json.load(eval_true_tables_f)

for plan in eval_inputs:
    truth_tables = truth_tables_set[plan]
    input_tables = eval_inputs[plan]

    ground_truths = {}
    for table in truth_tables:
        new_topics_dir = ground_truth_path + 'new_topics_[' + table + '].txt'
        with open(new_topics_dir, 'rb') as fp:
            topics_true = pickle.load(fp)
            ground_truths[table] = list(topics_true)

    print(plan)
    # pprint.pprint(reverse_dict(ground_truths))

    plan_output_topics = {}

    print('brute force')
    with open(brute_path + 'bruteforce_k5.json', 'r') as k5_f:
        table_topics_set_k5 = json.load(k5_f)
    with open(brute_path + 'bruteforce_k10.json', 'r') as k10_f:
        table_topics_set_k10 = json.load(k10_f)

    if plan in table_topics_set_k5:
        plan_output_topics = table_topics_set_k5[plan]
    elif plan in table_topics_set_k10:
        plan_output_topics = table_topics_set_k10[plan]

    # pprint.pprint(reverse_dict(plan_output_topics))

    precision, recall = compute_precision_and_recall(reverse_dict(ground_truths),[reverse_dict(plan_output_topics)])

    print('precision:', precision, 'recall:', recall)


    print('iterative')
    with open(iter_path + 'kb_file_v1_parks_10_'+plan+'.json', 'r') as f:
        table_matches_set = json.load(f)

    plan_output_topics = {}
    plan_output_topics[guiding_table] = []

    for topic in table_matches_set:
        topics_in_cluster = []
        tables_in_cluster = []

        for tar_tbl in table_matches_set[topic]['matches']:
            tar_topic = table_matches_set[topic]['matches'][tar_tbl]['attribute']

            topics_in_cluster.append(tar_topic)
            tables_in_cluster.append(tar_tbl)

        topics_in_cluster = list(set(topics_in_cluster))
        tables_in_cluster = list(set(tables_in_cluster))

        for tar_tbl in tables_in_cluster:
            if tar_tbl not in plan_output_topics:
                plan_output_topics[tar_tbl] = []

            plan_output_topics[tar_tbl] = list(set(plan_output_topics[tar_tbl] + topics_in_cluster))

    precision, recall = compute_precision_and_recall(reverse_dict(ground_truths), [reverse_dict(plan_output_topics)])

    print('precision:', precision, 'recall:', recall)


    print('data driven')
    with open(data_path+guiding_table+'_'+plan+'.json', 'r') as f:
        table_matches_set = json.load(f)

    new_topics = []
    for topic in table_matches_set:
        new_topics.append(table_matches_set[topic])

    new_topics = list(set(new_topics + list(table_matches_set.keys())))

    precision, recall = compute_precision_and_recall(reverse_dict(ground_truths), [reverse_dict({guiding_table: new_topics})])
    print('precision:', precision, 'recall:', recall)

    precision, recall = compute_precision_and_recall(reverse_dict({guiding_table: ground_truths[guiding_table]}),
                                                     [reverse_dict({guiding_table: new_topics})])
    print('guiding table precision:', precision, 'and recall:', recall)


    print('iterative and data driven')
    with open(iter_data_path + 'dataset_topics_v2_parks_10_'+plan+'.json', 'r') as f:
        plan_output_topics = json.load(f)

    precision, recall = compute_precision_and_recall(reverse_dict(ground_truths),[reverse_dict(plan_output_topics)])
    print('precision:', precision, 'recall:', recall)

    print('iterative and data driven (no checks)')

    fname = iter_data_noch_path + 'dataset_topics_v2_parks_10_'+plan+'.json'
    if not os.path.isfile(fname):
        continue
    with open(fname, 'r') as f:
        plan_output_topics = json.load(f)

    precision, recall = compute_precision_and_recall(reverse_dict(ground_truths),[reverse_dict(plan_output_topics)])
    print('precision:', precision, 'recall:', recall)


# for each attr in a table
# compare with another attr in a second table
# (use tfidf)
# find all attr-attr pairs that have a high score
# cluster attrs based on score
# for each cluster, figure out the best topic

import json

root = '/Users/haoran/Documents/thesis_schema_integration/'

tags_path = root+'inputs/datasource_and_tags.json'
schema_path = root+'inputs/schema_complete_list.json'

f = open(tags_path)
tag_data = json.load(f) # collaborative filtering

f = open(schema_path)
schema_data = json.load(f)

# target = ['aquatic hubs','drainage water bodies','park specimen trees', 'parks', 'park screen trees']
# target = ['park specimen trees']
# target = ['park specimen trees', 'park screen trees']
# target = ['aquatic hubs','drainage water bodies', 'park screen trees', 'park specimen trees']

# target =                               [
#             "park paths and trails",
#             "park natural areas",
#             "park specimen trees",
#             "park unimproved parkland",
#             "heritage sites",
#             "aquatic hubs",
#             "water fittings",
#             "park sports fields",
#             "road row requirements downtown"
#           ]
# target = [
# 'parks', 'heritage sites', 'water utility facilities', 'sanitary lift stations', 'drainage dyke infrastructure'
# ]
target = [
'parks', 'park outdoor recreation facilities', 'park sports fields', 'water assemblies', 'road row requirements downtown'
]
source = ['parks']

stats_path = root+'inputs/dataset_statistics/'

def reduce_attr_values(schema_path, schema_data):
    num_of_bins = 20
    num_of_bins_plus = 5
    for i, (dataset,dataset_schema) in enumerate(schema_data.items()):
        print('processing', dataset)

        for attr in dataset_schema:
            if attr['domain'] == 'coded_values' or attr['domain'] == 'coded_values_groupby':
                new_values = []
                new_values_plus = []
                len_vals = len(attr['coded_values'])
                median_intervals = len_vals // num_of_bins
                median_intervals_plus = len_vals // (num_of_bins*num_of_bins_plus)

                if num_of_bins_plus*num_of_bins > len_vals: continue     
                print('interval', median_intervals)

                datatype_not_int = 0
                not_ints = []
                for i,v in enumerate(attr['coded_values']):
                    if i % (median_intervals_plus) == 0:
                        new_values_plus.append(v)
                    if i % median_intervals == 0: 
                        new_values.append(v)
                        print('append', v)
                    elif i == len_vals: 
                        new_values.append(v)
                        print('append', v)

                    try:
                        # v = int(v)
                        v = float(v)
                    except:
                        datatype_not_int += 1
                        not_ints.append(v)

                if datatype_not_int / len_vals < 0.30:
                    attr['coded_values'] = new_values + not_ints
                    print('replace', new_values)
                else:
                    attr['coded_values'] = new_values_plus


    with open(schema_path, 'w') as fp:
        json.dump(schema_data, fp, sort_keys=True, indent=2)

import sys
path_lib = '/Users/haoran/Documents/neuralnet_hashing_and_similarity/tutorial_domain_hypothesis/'
sys.path.insert(0, path_lib)
import domain_hypothesis_v2 as dh

path_log = root+'notes/'
import logging
logging.basicConfig(filename=path_log + 'messages.log',level=logging.DEBUG)

instance_matching_output = root+'outputs/instance_matching_output2/'
import os

import sys
sys.path.insert(0, root)
from schema_matchers import matcher_name

from similarity.ngram import NGram    

import numpy as np
import pandas as pd

def find_datatype(src_vals):
    datatype_not_int = 0
    for val in src_vals:
        try:
            float(val)
        except:
            datatype_not_int += 1
    if datatype_not_int/max(len(src_vals),1) > 0.7:
        src_datatype = 'str' 
    else:
        src_datatype = 'num' 

    return src_datatype

import time
total_pre = 0
def pre_clustering(stats_path, source, target, instance_matching_output):

    t0 = time.time()

    twogram = NGram(2)
    threshold = 0.7
    weights = [0.0, 1.0]

    for src_table in source:
        src_path = stats_path + src_table + '.json'
        f = open(src_path)
        src_data = json.load(f)
        src_attrs = list(src_data.keys())

        for tar_table in target:
            print('-----')
            print(src_table, tar_table)

            tar_path = stats_path + tar_table + '.json'
            f = open(tar_path)
            tar_data = json.load(f)
            tar_attrs = list(tar_data.keys())

            sim_matrix = np.zeros((len(src_data), len(tar_data)))

            for i in range(len(src_attrs)):
                src_vals = src_data[src_attrs[i]]
                
                src_datatype = find_datatype(src_vals)

                for j in range(len(tar_attrs)):
                    tar_vals = tar_data[tar_attrs[j]]

                    tar_datatype = find_datatype(tar_vals)
                    print(src_attrs[i], tar_attrs[j])

                    U_set = 0.0
                    if src_datatype == 'str' and tar_datatype == 'str':

                        # n_a, n_b, D, n_D, t, n_t = dh.compute_sets(src_vals, tar_vals, threshold, matcher_name, twogram)
                        # U_set = dh.cdf(n_t, n_a, n_b, n_D)
                        pass
                    else:
                        U_set = 0.0

                    name_sim = matcher_name(src_attrs[i], tar_attrs[j], twogram)

                    print(U_set, name_sim)

                    if U_set > 1.0: U_set = 1.0
                    sim_matrix[i,j] = U_set*weights[0] + name_sim*weights[1] 

                    df_sim_matrix = pd.DataFrame(data=sim_matrix, columns=tar_attrs, index=src_attrs)

            filename = instance_matching_output + src_table + '/' 

            if not os.path.exists(filename):
                os.makedirs(filename)

            filename += '%s||%s.csv' % (src_table, tar_table)
            df_sim_matrix.to_csv(filename, sep=',', encoding='utf-8')
            msg = 'Matrix saved for src=%s tar=%s to %s'  %(src_table, tar_table, filename)
            logging.info(msg)

    t1 = time.time()
    total = t1 - t0
    total_pre = total
    print('preclustering time %s sec' % (total))

    return

import refine_kb_concepts as rkc
import pprint


def clustering(sim_matrix_path, threshold):
    merged_clusters = {}
    # no need to perform clustering, each src attr is a cluster

    t0 = time.time()

    for root, dirs, files in os.walk(sim_matrix_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.csv':
                continue
            output_df = pd.read_csv(root + '/' + file, index_col=0, header=0)
            tokens = filename.split('||')

            src_dataset = tokens[0]
            tar_dataset = tokens[1]
            if src_dataset == tar_dataset:
                continue

            columns = list(output_df.columns.values)
            if len(columns) == 0:
                print('ERROR empty target', tar_dataset)
                continue

            indexes = list(output_df.index.values)

            clusters = {}
            for index, row in output_df.iterrows():
                matches = []
                for col in columns:
                    if row[col] > threshold:
                        matches.append(col)
                clusters[index] = matches

            # print(clusters)

            if src_dataset not in merged_clusters: 
                merged_clusters[src_dataset] = {}
            merged_clusters[src_dataset][tar_dataset] = clusters

    # pprint.pprint(merged_clusters)
    tags = {}

    for i, (src_dataset, tar_datasets) in enumerate(merged_clusters.items()):
        merged = {}
        tags[src_dataset] = []
        for j, (tar_dataset, clusters) in enumerate(tar_datasets.items()):
            for k, (src_attr, tar_attrs) in enumerate(clusters.items()):
                if src_attr not in merged: merged[src_attr] = []
                if len(tar_attrs) != 0: 
                    for tar_attr in tar_attrs:
                        merged[src_attr].append((tar_dataset, tar_attr))

                tags[tar_dataset] = []

        print('merged: ', )
        pprint.pprint(merged)
        total_tags = []
        for _, (src_attr, tar_attrs) in enumerate(merged.items()):
            if len(tar_attrs) == 0: continue 
            group = [(src_dataset, src_attr)] + tar_attrs
            datasets = [ds for _, (ds, attr) in enumerate(group)]
            datasets = list(set(datasets))
            tag_candidates = {}
            for ds in datasets:
                # print(ds)
                tag_candidates[ds]=[tag['display_name'] for i, tag in enumerate(tag_data[ds]['tags'])] 
                for ds2 in datasets:
                    tags[ds2] += tag_candidates[ds]
                    tags[ds2] = list(set(tags[ds2]))
                    tags[ds2].sort()

                total_tags += tag_candidates[ds]
            # pprint.pprint(tag_candidates)

    t1 = time.time()
    total = t1 - t0
    print('clustering time %s sec' % (total))


    print('tags')
    pprint.pprint(tags)
    total_tags = list(set(total_tags))
    total_tags.sort()
    print('total tags')
    pprint.pprint(total_tags)


    print('preclustering time %s sec' % (total))
    print('clustering time %s sec' % (total))

    return tags

if __name__ == '__main__':
    pre_clustering(stats_path, source, target, instance_matching_output)
    threshold=0.6
    tags = clustering(instance_matching_output, threshold) # TODO: replace instance matching with schema matching

# given training set of attr to topic mapping
# use a combined score from many matching algorithms, find the best weights
# test with new attrs, store mapped topics
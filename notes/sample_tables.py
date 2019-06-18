import os 
import json
import pickle
import pprint

tables = {}
unique_topics = []
rootDir = '/Users/haoran/Documents/thesis_schema_integration/outputs/updated_topics/'
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    for fname in fileList:
        if '.txt' in fname and 'new_topic' in fname:
            # print('\t%s' % fname)
            fname_tmp = fname.split('[')[-1]
            fname_tmp = fname_tmp.split(']')[0]

            with open(dirName+fname, 'rb') as fp:
                topics_new = list(pickle.load(fp))
                unique_topics = unique_topics + topics_new
                topics_new.sort()
                tables[fname_tmp] = {'path':fname, 'new_topics': topics_new}

print(len(tables))

statistics_path = '/Users/haoran/Documents/thesis_schema_integration/inputs/datasource_and_tags.json'
schema_path = '/Users/haoran/Documents/thesis_schema_integration/inputs/schema_complete_list.json'
f = open(statistics_path)
tag_data = json.load(f)

f = open(schema_path)
schema_data = json.load(f)

attr_excl_list = ['shape', 'objectid', 'facilityid']

candidates = []
# choosing a candidate guiding table
for name in tables:
    attr_count = sum([1 for elem in schema_data[name] if elem not in attr_excl_list])
    
    topics = [topic_dict['display_name'] for topic_dict in tag_data[name]['tags']]
    topic_count = len(topics)
    groups = [grp_dict['display_name'] for grp_dict in tag_data[name]['groups']]

    tables[name]['old_topics'] = topics
    tables[name]['old_groups'] = groups

    if attr_count >= 5 and topic_count >= 5:
        candidates.append(name)

print(len(candidates))

table_topics_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/table_topics.json'

# pprint.pprint(tables)


unique_topics = list(set(unique_topics))
unique_topics.sort()
print(len(unique_topics))

unique_topics_ind = {}
for i, top in enumerate(unique_topics):
    unique_topics_ind[top] = i

for table in tables:

    topic_vec = [0]*len(unique_topics)
    for top in tables[table]['new_topics']:
        topic_vec[unique_topics_ind[top]] = 1

    tables[table]['topic_vec'] = topic_vec

    # if table in candidates: print(table, tables[table]['topic_vec'])


distance_matrix = {}
tables_sorted = list(tables.keys())
tables_sorted.sort()

sample_sizes = [5,10,20]
sample_mixes = [10,20,40]
for table in tables:

    print(table)
    distance_matrix[table] = [0]*len(tables)
    for i, table_i in enumerate(tables_sorted):
        overlaps = [i for i, j in zip(tables[table]['topic_vec'], tables[table_i]['topic_vec']) if i == j and i != 0 and j != 0]

        distance_matrix[table][i] = len(overlaps)

    # print(distance_matrix[table])

    max_in_matrix = [(tables_sorted[i], val) for i,val in enumerate(distance_matrix[table])]
    max_in_matrix = sorted(max_in_matrix, key=lambda x: x[1], reverse=True)
    # print(max_in_matrix)

    tables[table]['samples'] = {}
    for sample_size in sample_sizes:
    	sample = max_in_matrix[:sample_size+1]

    	# these are the highly related tables
    	tables[table]['samples'][sample_size] = [tabl[0] for tabl in sample]

    	# TODO then add some unrelated ones to the mix

with open(table_topics_path, 'w') as fp:
    json.dump(tables, fp, sort_keys=True, indent=2)
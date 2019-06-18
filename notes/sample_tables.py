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

    if attr_count >= 10 and topic_count >= 5:
        candidates.append(name)

print(candidates)

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

from random import sample 
sample_positive = [0,5,10,15,20,25,30,35,40]
sample_sizes = [10,20,40] 
num_guilding = 10

# 10:{(0,10),(5,5),(10,0)},        5 guiding
# 20:{(0,20),(5,15),(10,10),(15,5),(20,0)},        5 guiding
# 40:{(0,40),(5,35),(10,30),(15,25),(20,20),(25,15),(30,10),(35,5),(40,0)}        5 guiding
eval_plan = {}
for i in range(num_guilding):
    eval_plan[i] = sample(candidates,1)    # then eval for 10,20,40
    for item in eval_plan[i]:
        candidates.remove(item)
pprint.pprint(eval_plan)

for table in tables:

    # print(table)
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
        tables[table]['samples'][sample_size] = {}
        max_positive = sample_positive.index(sample_size)
        for i in range(max_positive+1):
            positive_count = sample_positive[i]
            negative_count = sample_size-(positive_count)
            # print((positive_count,negative_count))
            # these are the highly related tables
            positives = max_in_matrix[:positive_count]
            # then add some unrelated ones to the mix
            negatives = sample(max_in_matrix[positive_count:],negative_count)

            positives = [tabl[0] for tabl in positives]
            negatives = [tabl[0] for tabl in negatives]
            tables[table]['samples'][sample_size][str(positive_count)+'+'+str(negative_count)] = [positives,negatives]

with open(table_topics_path, 'w') as fp:
    json.dump(tables, fp, sort_keys=True, indent=2)

table_setup = {
'candidates': ['park playgrounds', 'park paths and trails', 'drainage devices', 'drainage flood control', 'park structures', 'drainage water bodies', 'water utility facilities', 'greenways', 'water sampling stations', 'drainage detention ponds', 'sanitary lift stations', 'trails and paths', 'sanitary flow system nodes', 'pay parking stations', 'water meters', 'drainage monitoring stations', 'drainage open channels', 'parks', 'bike routes', 'water assemblies', 'park outdoor recreation facilities', 'barriers'],
'guiding_tables' : {0: ['park outdoor recreation facilities'],
 1: ['park paths and trails'],
 2: ['parks'],
 3: ['pay parking stations'],
 4: ['park playgrounds'],
 5: ['drainage water bodies'],
 6: ['drainage monitoring stations'],
 7: ['sanitary lift stations'],
 8: ['sanitary flow system nodes'],
 9: ['greenways']},
'tables' : list(tables.keys())
}

table_setup_path = '/Users/haoran/Documents/thesis_schema_integration/outputs/table_setup.json'

with open(table_setup_path, 'w') as fp:
    json.dump(table_setup, fp, sort_keys=True, indent=2) 
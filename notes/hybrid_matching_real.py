import sys
path_lib = '/Users/haoran/Documents/neuralnet_hashing_and_similarity/tutorial_hybrid_flexmatcher/'
sys.path.insert(0, path_lib)
from flexmatcher import FlexMatcher
import pandas as pd
import json
import numpy as np
from pprint import pprint
import time

stats_path = '/Users/haoran/Documents/thesis_schema_integration/inputs/dataset_statistics/'

# 'parks' related tables as the training data, then test 'parks' for mapping its attrs to tags
# limitations: only as many tags as attrs, 
#     mappings not in training data are not seen at test time
#     tags set is not complete, 
#     how to discover the most useful word in an attribute,
#    data values are not always good representation of the attribute: some are bools, some are texts, some are nums, did not consult schema
#     can always just scan all words and collect all words similar to an existing topic

def get_top_values_for_attr(table_name, table_attrs):
    data = {}    
    with open(stats_path+name1+'.json') as f:
        data = json.load(f)

    data_vals = []
    not_in = []
    for attr in table_attrs:
        if attr not in data:
            not_in.append(attr)
            data_vals.append(['']*5)
            continue
        tbl_vals = data[attr]
        top5 = []
        sorted_tbl = sorted(tbl_vals.items(), key=lambda kv: kv[1])
        if len(sorted_tbl) < 5:
            pad_val = ''
            if len(sorted_tbl) != 0: 
                pad_val = sorted_tbl[0][0]
            sorted_tbl = sorted_tbl + [(pad_val, 0)]*(5-len(sorted_tbl))
        
        top5 = sorted_tbl[:5]
        top5 = [tup[0] for tup in top5]
        data_vals.append(top5)

        # print(attr)
        # print(top5)
    # print(not_in)
    return (data_vals)

def fill_values(name, tags, table):
    table_mapping = {table[0][i]:v  for i,v in enumerate(tags) }
    # print(table_mapping1)
    data_vals = get_top_values_for_attr(name, table[0])
    data_vals = np.array(data_vals)
    data_vals_inverse = np.transpose(data_vals)
    data_vals_inverse = data_vals_inverse.tolist()
    table = table + data_vals_inverse
    # pprint(table)
    return table, table_mapping

inputs = []

name1 = "park paths and trails"
tags1 = ['bike', 'network', 'corridor', 'type', 'facility', 'fencing', 'location', 'owners', 'park', 'trail', 'projects', 'status', 'trees']
table1 = [['bike infrastructure type', 'bike network', 'corridor name', 'corridor type', 'facilityid', 'fencing side1', 'location', 'owner', 'park ownership', 'prk trail service areas', 'project no', 'status', 'trees planted'],
]
inputs.append([name1, tags1, table1])

name2 = "park natural areas"
tags2 = ['land', 'urban', 'edges', 'facility', 'age', 'type', 'location', 'nature', 'park', 'owners', 'right of ways', 'service', 'trail']
table2 = [['adjcent land use rating', 'dedicated urban forest', 'exposed edge length m', 'facilityid', 'forest age class', 'forest type', 'location', 'natural area type', 'park name', 'park ownership', 'right of way', 'service', 'trail present'],
]
inputs.append([name2, tags2, table2])

name3 = 'park specimen trees'
tags3 = ['donation', 'heritage', 'location', 'park', 'owners', 'status', 'trees', 'size', 'type', 'water']	# 'heights' is not mapped to 'height m' because of different meanings	# park map to park, but missed parks, so all other tables also need to map to park
table3 = [['gift donation', 'heritage significant tree', 'location', 'park', 'park ownership', 'status', 'tree', 'tree size classification', 'tree type', 'tree watering'],
]
inputs.append([name3, tags3, table3])

name4 = 'park unimproved parkland'
tags4 = ['house', 'location', 'type', 'owners', 'code', 'service', 'status', 'surface'] 
table4 = [['house no', 'location', 'park area type2', 'park ownership', 'road code', 'service level', 'status', 'surface type'],
]
inputs.append([name4, tags4, table4])

name5 = 'park outdoor recreation facilities'
tags5 = ['community', 'type', 'facility', 'fields', 'light', 'location', 'outdoor', 'park', 'owners', 'use', 'services', 'status', 'surface'] # 'court type', which word is more important? court or type?
table5 = [['community', 'court type', 'facilityid', 'field type', 'lighting system', 'location', 'outdoor rec fac type2', 'park', 'park ownership', 'primary use', 'service level', 'status', 'surface type'],
]
inputs.append([name5, tags5, table5])

name6 = 'park sports fields'
tags6 = ['school', 'community', 'facility', 'location', 'park', 'owners', 'use', 'fields', 'status', 'surface']	# secondary as in 'secondary school' or 'secondary use' 
table6 = [['adjacent to school', 'community', 'facilityid', 'location', 'park', 'park ownership', 'primary use', 'sports fields type2', 'status', 'surface type'],
]
inputs.append([name6, tags6, table6])

name7 = 'park structures'
tags7 = ['bench', 'dimensions', 'donation', 'location', 'type', 'park', 'feature', 'structure']	# training data created without looking at testing data what possible mappings, created by looking at the data values of each attribute and all tables that contain the topic
table7 = [['bench style', 'dimension m2', 'gift donation', 'location', 'material type', 'park ownership', 'special features', 'structure type'],
]
inputs.append([name7, tags7, table7])

name8 = 'trails and paths'
tags8 = ['corridor', 'type', 'location', 'owners', 'route', 'status']	# important tags are missing, such as 'material' (for attr 'material') and 'maintainence' (or 'responsonilty') (for attr 'maintainence responsibility'),	# 'route type2' mapped to 'route' instead of 'type' because values such as 'Nature Trail' is more closely related to 'route'
table8 = [['corridor name', 'corridor type', 'operating location', 'owner', 'route type', 'status'],
]
inputs.append([name8, tags8, table8])

name9 = 'walking routes'	# rule: try to map attributes that have values instead of empty
tags9 = ['network', 'corridor', 'type', 'facility', 'location', 'status']		# these tags are also found in the gold standard
table9 = [['bike network', 'corridor name', 'corridor type', 'facilityid', 'location', 'status'],
]
inputs.append([name9, tags9, table9])

name10 = "park screen trees"
tags10 = ['facility', 'location', 'trees', 'park', 'owners', ]
table10 = [['facilityid', 'location', 'number of trees', 'park name', 'park ownership'],
]
inputs.append([name10, tags10, table10])

schema_list = []
mapping_list = []

def create_dataframe_and_mapping(name1, tags1, table1, schema_list, mapping_list):
    table1, table_mapping1 = fill_values(name1, tags1, table1)
    header = table1.pop(0)
    table1_pd = pd.DataFrame(table1, columns=header)
    # print(table1_pd)
    schema_list.append(table1_pd)
    mapping_list.append(table_mapping1)
    return schema_list, mapping_list

start = time.time()

for item in inputs:
	schema_list, mapping_list = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list, mapping_list)

# table2, table_mapping2 = fill_values(name2, tags2, table2)
# header = table2.pop(0)
# table2_pd = pd.DataFrame(table2, columns=header)

# table10, table_mapping10 = fill_values(name10, tags10, table10)
# header = table10.pop(0)
# table10_pd = pd.DataFrame(table10, columns=header)
 
# tables = [  ["park paths and trails",table1],
#             ["park natural areas",table2],
#             ["park specimen trees",table3],
#             ["park unimproved parkland",table4],
#             ["park outdoor recreation facilities",table5],
#             ["park sports fields",table6],
#             ["park structures",table7],
#             ["trails and paths",table8],
#             ["walking routes",table9],
#             ["park screen trees",table10],    ]

# schema_list = [table1_pd, table2_pd, table10_pd]
# mapping_list = [table_mapping1, table_mapping2, table_mapping10]

fm = FlexMatcher(schema_list, mapping_list, sample_size=100)
fm.train()  

name100 = "parks"
table100 = [['comments', 'description', 'facilityid', 'house no', 'location', 'maintenance responsibility', 'map page', 'operational responsibility', 'park name', 'park ownership', 'road code', 'service classification', 'status', 'work assignment'],
]
table100, _ = fill_values(name100, [], table100)
header = table100.pop(0)
table100_pd = pd.DataFrame(table100, columns=header)

# # 10+0 
# predicted_mapping = fm.make_prediction(table100_pd)
# pprint(predicted_mapping)

# end = time.time()

# import json
# with open('parks_10+0.json', 'w', encoding='utf-8') as f:
#     json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)

inputs2 = []

name11 = "drainage dyke infrastructure"
tags11 = ['protection','infrastructure','station','sites','size','water']
table11 = [['corrosion protection','infrastructure type','public and private pump station','site id','size','watercourse'],
]
inputs2.append([name11, tags11, table11])

name12 = "sanitary lift stations"
tags12 = ['facility','location','owners','projects','status']
table12 = [['facilityid','location','owner','project no','status'],
]
inputs2.append([name12, tags12, table12])

name13 = "water utility facilities"
tags13 = ['type','facility','location','owners','projects','status']	# facilityid in 'water' domain is different from 'park' domain
table13 = [['facility type2','facilityid','location','owner','project no','status'],
]
inputs2.append([name13, tags13, table13])

name14 = "water valves"
tags14 = ['protection','disaster','facility','location','node','owners','projects','status','valve']	# if values are just 'Yes' or 'No', then attr name is more meaningful
table14 = [['corrosion protected','disaster control','facilityid','location','node no','owner','project no','status','valve type2'],
]	# if examine both attr name and values, what about a decision tree, might work better than just weight and add
inputs2.append([name14, tags14, table14])

name15 = "water meters"
tags15 = ['facility','folio','house','image','status','street']	# 'account' does not map to 'accounting'
table15 = [['facilityid','folio','house no','image','status','street'],
]
inputs2.append([name15, tags15, table15])

name16 = "sanitary nodes"
tags16 = ['facility','location','node','owners','projects','status','structure']
table16 = [['facilityid','location','node type2','owner','project no','status','structure'],
]
inputs2.append([name16, tags16, table16])

name17 = "heritage routes"
tags17 = ['heritage','routes','type']	# 'address' not a topic?	# 'type' is a very ambiguous word 
table17 = [['heritage value','route type','type'],
]
inputs2.append([name17, tags17, table17])

name18 = "water assemblies"
tags18 = ['air','drainage','facility','location','node','owners','projects','status','valve']	# but these tags are used so many times, what about new tags that hasn't been discovered?
table18 = [['air valve type','drain system','facilityid','location','node no','owner','project no','status','valve type'],
]
inputs2.append([name18, tags18, table18])

name19 = "water fittings",
tags19 = ['depth','facility','fitting','location','owners','projects','elevation','status']	# here 'elevation' is a good topic, although it was used for other purposes describing land
table19 = [['depth','facilityid','fitting type2','location','owner','project no','rim elevation','status'],
]
inputs2.append([name19, tags19, table19])

name20 = "road row requirements downtown"
tags20 = ['road']
table20 = [['road width'],
]
inputs2.append([name20, tags20, table20])

name21 = "aquatic hubs"
tags21 = ['hubs','type']
table21 = [['name','type'],
]
inputs2.append([name21, tags21, table21])

# start2 = time.time()

# schema_list2 = []
# mapping_list2 = []

# for item in inputs2:
# 	schema_list2, mapping_list2 = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list2, mapping_list2)

# fm = FlexMatcher(schema_list2, mapping_list2, sample_size=100)
# fm.train()  

# # 0+10 
# predicted_mapping = fm.make_prediction(table100_pd)
# pprint(predicted_mapping)

# end2 = time.time()

# with open('parks_1+9.json', 'w', encoding='utf-8') as f:
#     json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)

inputs3 = [inputs[0], inputs[1], inputs[2], inputs[3], inputs2[8], inputs2[9], inputs2[10], inputs2[3], inputs2[6]]  

# start3 = time.time()

# schema_list3 = []
# mapping_list3 = []

# for item in inputs3:
# 	schema_list3, mapping_list3 = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list3, mapping_list3)

# fm = FlexMatcher(schema_list3, mapping_list3, sample_size=100)
# fm.train()  

# predicted_mapping = fm.make_prediction(table100_pd)
# pprint(predicted_mapping)

# end3 = time.time()

# with open('parks_5+5.json', 'w', encoding='utf-8') as f:
#     json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)


# print(end - start)
# print(end2 - start2)
# print(end3 - start3)



# parks, park outdoor recreation facilities, park sports fields; water assemblies, road row requirements downtown
inputs4 = [inputs[4], inputs[5], inputs2[7], inputs2[9]]  

start4 = time.time()

schema_list4 = []
mapping_list4 = []

# for item in inputs4:
# 	schema_list4, mapping_list4 = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list4, mapping_list4)

# fm = FlexMatcher(schema_list4, mapping_list4, sample_size=100)
# fm.train()  

# # 5+5 
# predicted_mapping = fm.make_prediction(table100_pd)
# pprint(predicted_mapping)

# end4 = time.time()

# with open('parks_3+2.json', 'w', encoding='utf-8') as f:
#     json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)

# print(end4 - start4)	# 0.7710871696472168


# parks; water utility facilities, sanitary lift stations, drainage dyke infrastructure, water meters
inputs5 = [inputs2[1], inputs2[2], inputs2[0], inputs2[4], inputs[7]]  

# start5 = time.time()

# schema_list5 = []
# mapping_list5 = []

# for item in inputs5:
# 	schema_list5, mapping_list5 = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list5, mapping_list5)


# fm = FlexMatcher(schema_list5, mapping_list5, sample_size=100)
# fm.train()  

# predicted_mapping = fm.make_prediction(table100_pd)
# pprint(predicted_mapping)

# end5 = time.time()

# with open('parks_1+4.json', 'w', encoding='utf-8') as f:
#     json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)

# print(end5 - start5)	# 0.7553079128265381

# 'park specimen trees', 'parks', 'park screen trees', 'park outdoor recreation facilities', "park structures"
inputs6 = [inputs[2], inputs[9], inputs[4], inputs[6]]  

start6 = time.time()

schema_list6 = []
mapping_list6 = []

for item in inputs6:
	schema_list6, mapping_list6 = create_dataframe_and_mapping(item[0],item[1],item[2], schema_list6, mapping_list6)


fm = FlexMatcher(schema_list6, mapping_list6, sample_size=100)
fm.train()  

predicted_mapping = fm.make_prediction(table100_pd)
pprint(predicted_mapping)

end6 = time.time()

with open('parks_5+0.json', 'w', encoding='utf-8') as f:
    json.dump(predicted_mapping, f, ensure_ascii=False, indent=4)

print(end6 - start6)	# 0.80147385597229
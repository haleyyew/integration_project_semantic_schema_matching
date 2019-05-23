from os import listdir
from os.path import isfile, join
import json
import pprint

dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/enriched_topics/'
onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and '.json' in f]

onlyfiles_open = []

onlyfiles.sort()

for f in onlyfiles:
	f_p = open(join(dir, f), 'r')
	f_o = json.load(f_p)


	print('len', len(f_o[1]))

	contexts = {}
	if len(f_o[1]) == 1:
		for syn in f_o[1][0][1]:
			print(syn)
			print(len(f_o[1][0][1][syn]))
			if len(f_o[1][0][1][syn]['overlap']) > 0:

				contexts[syn] = f_o[1][0][1][syn]['overlap'] 		

	if len(f_o[1]) > 1:
		for i in range(len(f_o[1])):
			for j in range(len(f_o[1][i])):
				for syn in f_o[1][i][j]:
					# print(syn)
					# print(len(f_o[1][i][j][syn]))
					if 'overlap' not in f_o[1][i][j][syn]:
						continue
					if len(f_o[1][i][j][syn]['overlap']) > 0:
						if syn in contexts:
							contexts[syn] = list(set(contexts[syn].extend(f_o[1][i][j][syn]['overlap'] )))
						else:
							contexts[syn] = f_o[1][i][j][syn]['overlap'] 

	onlyfiles_open.append(contexts)

pprint.pprint(onlyfiles)
pprint.pprint(onlyfiles_open)

onlyfiles_re = []
for f in onlyfiles:
	f = f.replace('[', '')
	f = f.replace(']', '')
	f = f.replace('.json', '')
	onlyfiles_re.append(f)

pprint.pprint(onlyfiles_re)

changed = ''

dataset_topics = {}
for i in range(len(onlyfiles_re)):
	dataset_topic = onlyfiles_re[i]
	split_topic = dataset_topic.split('_')

	if changed != split_topic[0]:
		dataset_topics[split_topic[0]] = {}
		changed = split_topic[0]
	dataset_topics[split_topic[0]][split_topic[1]] = onlyfiles_open[i]

pprint.pprint(dataset_topics)

with open(join(dir, 'dataset_topics_enriched.json'), 'w') as fp:
    json.dump(dataset_topics, fp, sort_keys=True, indent=2)

# remember to move the output file out of dir
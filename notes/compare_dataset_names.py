path = '/Users/haoran/Documents/thesis_schema_integration/notes/impl_plan0504.txt'
fo = open(path, "r")

lines = fo.readlines()

# if len(lines) > 0:
#     print(lines[0])

sections = [('-start saved-', '-end saved-'), ('-start community services-', '-end community services-'), ('-start recreation and culture-', '-end recreation and culture-'), ('-start environmental services-', '-end environmental services-'), ('-start infrastructure-', '-end infrastructure-'), ('-start transportation-', '-end transportation-')]

containers = []

i = 0

for sec in sections:
    sec_start = sec[0]
    sec_end = sec[1]
    while lines[i].strip() != sec_start:
        # print(lines[i])
        i += 1

    i += 1

    container = []
    while lines[i].strip() != sec_end:
        # print(lines[i])
        line = lines[i].strip()
        line = line.split('\t')
        # print(line)
        if len(line) > 1:
            line = line[1]
        line = ''.join(line)
        line = line.replace('"', '')
        container.append(line)
        i += 1

    i += 1

    containers.append(container)

fo.close()

import json
statistics_path = '/Users/haoran/Documents/thesis_schema_integration/inputs/datasource_and_tags.json'
schema_path = '/Users/haoran/Documents/thesis_schema_integration/inputs/schema_complete_list.json'
f = open(statistics_path)
tag_data = json.load(f)

f = open(schema_path)
schema_data = json.load(f)

exclusion_list = ['contours 1m', 'contours 5m']
attr_excl_list = ['shape', 'objectid', 'facilityid']

saved = containers[0]
print(saved)
all_datasets = []
unique_batches = []

def check_metadata(name):
    
    if name not in tag_data:
        return False

    if name not in schema_data:
        return False
        
    if name in exclusion_list:
        return False

    if len(schema_data[name]) == 0:
        return False

    attr_count = sum([1 for elem in schema_data[name] if elem not in attr_excl_list])

    if attr_count == 0:
        return False

    return True

for i in range(1, len(containers)):
    container = list(set(containers[i]) - set(saved))

    # print(container)

    container = list(set(container) - set(all_datasets))
    container = [name for name in container if check_metadata(name)]

    all_datasets.extend(container)
    unique_batches.append(container)


    print(sections[i])
    print(container)
    print(len(container))
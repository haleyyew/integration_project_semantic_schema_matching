import zipfile
import os
import shutil
import json
import csv
import copy

class DataModel(object):
    def __init__(self):
        self.datasets = {}
        self.tags = []
        # self.tag_map = {}
        self.groups = []

    def add_dataset(self, name, dataset):
        self.datasets[name] = dataset

    def print_data_model(self):
        print('Data Model:')
        print('tags: ' + str(self.tags))
        print('groups: ' + str(self.groups))
        print('-----')
        for key in  self.datasets:
            dataset = self.datasets[key]
            print_data_instance(dataset)
            print('-----')

class DataInstance(object):
    def __init__(self):
        self.name = ''
        self.notes = ''
        self.tags = []
        self.groups = []
        self.group_descriptions = []
        self.resources = []

    def set_name(self, name):
        self.name = name

    def add_resource(self, resource):
        self.resources.append(resource)

    def set_metadata(self, notes, tags, groups, group_descriptions):
        self.notes = notes
        self.tags = tags
        self.groups = groups
        self.group_descriptions = group_descriptions

def print_data_instance(data_instance):
    print('Data Instance:')
    print('name: ' + data_instance.name)
    print('notes: ' + data_instance.notes)
    print('tags: ' + str(data_instance.tags))
    print('groups: {')
    for i in range(len(data_instance.groups)):
        group = data_instance.groups[i]
        group_description = data_instance.group_descriptions[i]
        print('\t' + '(' + str(group) + ', ' + group_description + ')')
    print('}')

    print('resources: {')
    for i in range(len(data_instance.resources)):
        resource = data_instance.resources[i]
        print('\t' + 'format: ' + resource['format'])
        # for i in range(len(resource['data'])):
        for i in range(10):
            value = resource['data'][i]
            print('\t' + '\t' + 'value: ' + str(value))

    print('}')


    return

def create_tags(tags_json, data_model_tags):
    tags = []
    for tag in tags_json:
        index = -1
        try:
            tag_name = tag['name']
            index = data_model_tags.index[tag_name]
        except Exception as e:
            print('error: create_tags ' + str(e))
            data_model_tags.append(tag['name'])
            index = len(data_model_tags) - 1

        if len(tags) <= index:
            grow_len = index - (len(tags) - 1)
            for i in range(grow_len):
                tags.append(0)

        tags[index] = 1

    return tags

def create_groups(groups_json, data_model_groups):
    groups = []
    groups_descriptions = []
    for group in groups_json:
        index = -1
        try:
            group_name = group['display_name']
            index = data_model_groups.index[group_name]
        except Exception as e:
            print('error: create_groups ' + str(e))
            data_model_groups.append(group['display_name'])
            index = len(data_model_groups) - 1

        if len(groups) <= index:
            grow_len = index - (len(groups) - 1)
            for i in range(grow_len):
                groups.append(0)
                groups_descriptions.append('')
        groups[index] = 1
        groups_descriptions[index] = group['description']

    return groups, groups_descriptions

def create_resource_from_csv(resource_csv, data_instance, data_model):
    resource = {}
    resource['format'] = 'csv'
    resource['data'] = resource_csv

    return resource

def create_resource_from_json(resource_json, data_instance, data_model):
    resource = {}
    resource['format'] = 'json'
    resource['data'] = []

    for element in resource_json['features']:
        row = copy.deepcopy(element['properties'])
        row['geometry_type'] = element['geometry']['type']
        resource['data'].append(row)

    return resource

def unzip_and_rename():
    # zip_ref = zipfile.ZipFile('../thesis_project_dataset/parks/parks_JSON.zip', 'r')
    # zip_ref.extractall('../thesis_project_dataset/parks')
    # zip_ref.close()

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            # print(len(path) * '---', file)
            # print(root + '/' + file)
            filename, file_extension = os.path.splitext(file)
            if file == '.DS_Store':
                continue
            # 	print('skip')
            # print(file_extension)
            try:
                if file_extension == '.zip':
                    # print(root + '/' + file)
                    zip_ref = zipfile.ZipFile(root + '/' + file, 'r')
                    zip_ref.extractall(root)
                    zip_ref.close()
                if file_extension == '':
                    # print(root + '/' + file)
                    shutil.copyfile(root + '/' + file, root + '/' + file + '.csv')
            except Exception as e:
                print('error: unzip_and_rename ' + str(e))


def parse_models():

    data_to_parse = ['important-trees', 'parks']
    data_model = DataModel()

    with open('./downloadResourceURL.json', 'r') as f:

        data = json.load(f)
        for key in data:
            if key in data_to_parse:
                data_instance = DataInstance()

                # print(key)
                value = data[key]
                data_instance.set_name(key)
                tags = create_tags(value['tags'], data_model.tags)
                groups, group_descriptions = create_groups(value['groups'], data_model.groups)
                data_instance.set_metadata(value['notes'], tags, groups, group_descriptions)

                data_model.add_dataset(key, data_instance)

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        curr_dir_path = root.split("/")
        curr_dir_name = curr_dir_path[-1]

        if curr_dir_name in data_to_parse:
            for file in files:
                filename, file_extension = os.path.splitext(file)
                try:
                    if file_extension == '.json':
                        print(curr_dir_name + '/' + file)
                        with open(root + '/' + file, mode='r') as json_file:
                            json_data = json.load(json_file)



                            dataset_name = curr_dir_name
                            data_inst = data_model.datasets[dataset_name]

                            resource = create_resource_from_json(json_data, data_inst, data_model)
                            data_inst.add_resource(resource)




                    elif file_extension == '.csv':
                        print(curr_dir_name + '/' + file)
                        csv_data = []
                        with open(root + '/' + file, mode='r', encoding='unicode_escape') as csv_file:
                            csv_reader = csv.DictReader(csv_file)
                            for row in csv_reader:
                                csv_data.append(row)

                        dataset_name = curr_dir_name
                        data_inst = data_model.datasets[dataset_name]

                        resource = create_resource_from_csv(csv_data, data_inst, data_model)
                        data_inst.add_resource(resource)

                    else:
                        pass

                except Exception as e:
                    print('error: parse_models ' + str(e))
        else:
            continue

    data_model.print_data_model()


if __name__ == "__main__":
    # unzip_and_rename()
    parse_models()
    pass
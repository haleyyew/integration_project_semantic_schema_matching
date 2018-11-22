import zipfile
import os
import shutil
import json
import csv
import copy
import asyncio

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
            print('note: create_tags ' + str(e))
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
            print('note: create_groups ' + str(e))
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

    # cnt = 0
    for element in resource_json['features']:
        row = copy.deepcopy(element['properties'])
        row['geometry_type'] = element['geometry']['type']
        resource['data'].append(row)

        # if cnt < 10:
        #     print(resource['data'][-1])
        # cnt += 1

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

    data_to_parse = ['important-trees', 'parks', 'park-specimen-trees']
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


    added_datasource = {}

    for root, dirs, files in os.walk("../thesis_project_dataset"):
        curr_dir_path = root.split("/")
        curr_dir_name = curr_dir_path[-1]

        if curr_dir_name in data_to_parse:
            for file in files:

                filename, file_extension = os.path.splitext(file)

                if curr_dir_name in added_datasource:
                    print('break', curr_dir_name)
                    # print(added_datasource)
                    break
                if file_extension == '.json' or file_extension == '.csv':
                    added_datasource[curr_dir_name] = 1
                    # print('added', curr_dir_name, 'during', file)




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

    # data_model.print_data_model()
    # TODO: json doesn't print

    return data_model

class KnowledgeBase:
    def __init__(self):
        pass

from langdetect import detect
from googletrans import Translator

def translate_to_english(str):
    translator = Translator()
    try:
        translated = translator.translate(str, dest='en')
    except Exception:
        return {'2.origin': str}

    return {'1.src': translated.src, '2.origin': translated.origin, '3.dest': translated.dest, '4.text': translated.text}

    # print(str)
    #
    # try:
    #     lang = detect(str)
    # except Exception:
    #     print('!! error', str)
    #     return (False, 'NULL')
    #
    # translator = Translator()
    #
    # if lang == 'en':
    #     return (True, str)
    # else:
    #     translated = translator.translate(str, dest='en')
    #     print('    false', lang, ' - ', str, ' - ', translated)
    #     return (False, translated)

import json
def collect_concepts():
    list_of_tags = []
    list_of_groups = []
    with open('./metadata/tag_list.json', 'r') as f:
        data = json.load(f)
        list_of_tags = [translate_to_english(key) for key in data['result']]

        # cnt = 0
        # for key in data['result']:
        #     list_of_tags.append(translate_to_english(key))
        #     cnt += 1
        #     if cnt > 10:
        #         break

    with open('./metadata/group_list.json', 'r') as f:
        data = json.load(f)
        list_of_groups = [translate_to_english(key) for key in data['result']]

        # cnt = 0
        # for key in data['result']:
        #     list_of_groups.append(translate_to_english(key))
        #     cnt += 1
        #     if cnt > 10:
        #         break

import toy.data as td
def collect_concepts_beta():
    tags = []
    tags.extend([(tag.lower(), 'PARKS', 'tag') for tag in td.PARKS_metadata['tags']])
    tags.extend([(tag.lower(), 'IMPORTANTTREES', 'tag') for tag in td.IMPORTANTTREES_metadata['tags']])
    tags.extend([(tag.lower(), 'PARKSPECIMENTREES', 'tag') for tag in td.PARKSPECIMENTREES_metadata['tags']])

    tags.extend([(td.PARKS_metadata['category'].lower(), 'PARKS', 'category')])
    tags.extend([(td.IMPORTANTTREES_metadata['category'].lower(), 'IMPORTANTTREES', 'category')])
    tags.extend([(td.PARKSPECIMENTREES_metadata['category'].lower(), 'PARKSPECIMENTREES', 'category')])

    tags_set = {}
    for tag_tuple in tags:
        if tag_tuple[0] in tags_set:
            sources_list = tags_set[tag_tuple[0]]
            sources_list.append({'source_name': tag_tuple[1], 'meta_type': tag_tuple[2]})
            # update probability 1/len(sources_list)
        else:
            tags_set[tag_tuple[0]] = [{'source_name': tag_tuple[1], 'meta_type': tag_tuple[2]}]

    return tags_set

def parse_metadata(file):

    metadata = []


    with open(file, 'r') as f:
        data = json.load(f)


        for item in data['body']['fields']:
            attr = {}
            attr['name'] = item['name'].lower()
            if item['alias'] == item['name']:
                attr['alias'] = None
            else:
                attr['alias'] = item['alias'].lower()
            try:
                if item['domain'] == None:
                    attr['domain'] = None
                elif item['domain']['type'] == 'codedValue':
                    attr['domain'] = 'coded_values'
                    attr['coded_values'] = [value['name'] for value in item['domain']['codedValues']]
            except Exception as e:
                print('error', 'parse_metadata domain', file, e)

            try:
                attr['data_type'] = item['type']
            except Exception as e:
                print('error', 'parse_metadata type', file, e)

            metadata.append(attr)

    return metadata

if __name__ == "__main__":
    # unzip_and_rename()
    data_model = parse_models()

    pass
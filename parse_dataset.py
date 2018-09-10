import zipfile
import os
import shutil
import json
import csv

class DataModel:
    pass

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
                print('error: ' + str(e))


def parse_models():

    data_to_parse = ['important-trees', 'parks']

    with open('./downloadResourceURL.json', 'r') as f:
        data = json.load(f)
        for key in data:
            if key in data_to_parse:
                print(key)
                value = data[key]
                

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



                    elif file_extension == '.csv':
                        print(curr_dir_name + '/' + file)
                        csv_data = []
                        with open(root + '/' + file, mode='r', encoding='unicode_escape') as csv_file:
                            csv_reader = csv.DictReader(csv_file)
                            for row in csv_reader:
                                csv_data.append(row)


                    else:
                        pass

                except Exception as e:
                    print('error: ' + str(e))
        else:
            continue


if __name__ == "__main__":
    # unzip_and_rename()
    parse_models()
    pass
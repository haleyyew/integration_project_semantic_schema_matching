import os
import sys

scriptpath = "./parse_dataset.py"
sys.path.append(os.path.abspath(scriptpath))
import parse_dataset

class GraphicalModel(object):
    def __init__(self):
        self.nodes = []
        self.edges = []

def build_graph(data_model):
    for key in data_model.datasets:
        data_instance = data_model.datasets[key]
        print('Data Instance: ' + key)

        for resource in data_instance.resources:
            print(resource['format'])

            data = resource['data']
            first_row = data[0]
            for attribute_name in first_row:
                print('\t' + attribute_name)
    return

if __name__ == "__main__":
    data_model = parse_dataset.parse_models()
    build_graph(data_model)

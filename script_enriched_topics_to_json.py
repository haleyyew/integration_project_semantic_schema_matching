from os import listdir
from os.path import isfile, join
import json
import pprint



def enriched_topics_json(dir, dir_out, dataset_topics_dir, dataset_topics_extended_dir):


    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and '.json' in f]

    onlyfiles_open = []
    onlyfiles_open_extended = []

    onlyfiles.sort()

    for f in onlyfiles:
        f_p = open(join(dir, f), 'r')
        f_o = json.load(f_p)


        print('len', len(f_o[1]))

        contexts = {}
        contexts_extended = {}
        if len(f_o[1]) == 1:
            for syn in f_o[1][0][1]:
                print(syn)
                print(len(f_o[1][0][1][syn]))

                contexts_extended[syn] = {}
                contexts_extended[syn]['source_words'] = f_o[1][0][1][syn]['source_words']
                contexts_extended[syn]['target_words'] = f_o[1][0][1][syn]['target_words']

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

                        contexts_extended[syn] = {}
                        contexts_extended[syn]['source_words'] = f_o[1][i][j][syn]['source_words']
                        contexts_extended[syn]['target_words'] = f_o[1][i][j][syn]['target_words']

                        if len(f_o[1][i][j][syn]['overlap']) > 0:
                            if syn in contexts:
                                contexts[syn] = list(set(contexts[syn].extend(f_o[1][i][j][syn]['overlap'] )))
                            else:
                                contexts[syn] = f_o[1][i][j][syn]['overlap']

        # TODO sort by num of overlaps
        onlyfiles_open.append(contexts)
        onlyfiles_open_extended.append(contexts_extended)

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
    dataset_topics_extended = {}
    for i in range(len(onlyfiles_re)):
        dataset_topic = onlyfiles_re[i]
        split_topic = dataset_topic.split('_')

        if changed != split_topic[0]:
            dataset_topics[split_topic[0]] = {}
            dataset_topics_extended[split_topic[0]] = {}
            changed = split_topic[0]
        dataset_topics[split_topic[0]][split_topic[1]] = onlyfiles_open[i]
        dataset_topics_extended[split_topic[0]][split_topic[1]] = onlyfiles_open_extended[i]

    pprint.pprint(dataset_topics)


    with open(join(dir_out, dataset_topics_dir), 'w') as fp:
        json.dump(dataset_topics, fp, sort_keys=True, indent=2)

    with open(join(dir_out, dataset_topics_extended_dir), 'w') as fp:
        json.dump(dataset_topics_extended, fp, sort_keys=True, indent=2)

    # TODO remember to move the output file out of dir
    return


def one_full_run():
    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/enriched_topics/'
    dir_out = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    dataset_topics_dir = 'dataset_topics_enriched.json'
    dataset_topics_extended_dir = 'dataset_topics_enriched_extended.json'
    enriched_topics_json(dir, dir_out, dataset_topics_dir, dataset_topics_extended_dir)

    attrs_dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/attributes_enriched/'
    dataset_attrs_dir = 'dataset_attrs_enriched.json'
    dataset_attrs_extended_dir = 'dataset_attrs_enriched_extended.json'
    enriched_topics_json(attrs_dir, dir_out, dataset_attrs_dir, dataset_attrs_extended_dir)

if __name__ == "__main__":
    one_full_run()
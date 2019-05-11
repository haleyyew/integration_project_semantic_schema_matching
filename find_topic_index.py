import pickle
import argparse



if __name__ == '__main__':
    dir = '/Users/haoran/Documents/thesis_schema_integration/outputs/'
    with open(dir+'topics.txt', 'rb') as fp:
        topics_new = pickle.load(fp)

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--s', type=str, help='A required string positional argument')
    args = parser.parse_args()

    print(args.s)
    index_of = topics_new.index(args.s)
    print(index_of)

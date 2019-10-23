# sftp -i "/Users/haoran/Documents/thesis_schema_integration/notes/word2vec.pem" ubuntu@35.165.28.229
# put fasttext_wordvec_server_tmp.py
# put fasttext_wordvec_server.py
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
# gunzip cc.en.300.bin.gz
# sudo add-apt-repository universe
# sudo apt update
# sudo apt install python3-pip
# pip3 install --upgrade gensim
# pip3 install Flask
# pip3 install flask flask-jsonpify flask-sqlalchemy flask-restful
# pip3 install flask-cors --upgrade

from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

import time
t0 = time.time()

from gensim.test.utils import datapath
cap_path = datapath("/home/haoran/cc.en.300.bin")
from gensim.models.wrappers import FastText
wv = FastText.load_fasttext_format(cap_path)

t1 = time.time()
CORS(app)

print('app started')
print('time to load:', t1-t0)

def most_similar(word):
	return wv.most_similar(word)
     
def similarity(word1, word2):
    return wv.similarity(word1, word2)
    
from ast import literal_eval
import numpy as np
def vector(word):
    a = wv[word]
    a_str = ''.join(np.array2string(a, separator=',').splitlines())
    return a_str
    
class MostSimilar(Resource):
    def get(self, input):
        output = most_similar(input)
        return str(output)

class Similarity(Resource):
    def get(self, input):
        splt = input.split('__')
        if len(splt) != 2: return 'ERROR'
        input1 = splt[0]
        input2 = splt[1]
        inp1_splt = input1.split()
        inp2_splt = input2.split()
        # 2 gram
        for i, item in enumerate(inp1_splt):
            if i == len(inp1_splt)-1: inp1_splt.pop()
            else: inp1_splt[i] = item + ' ' + inp1_splt[i+1]
        for i, item in enumerate(inp2_splt):
            if i == len(inp2_splt)-1: inp2_splt.pop()
            else: inp2_splt[i] = item + ' ' + inp2_splt[i+1]

        print(inp1_splt)
        print(inp2_splt)

        max_output = 0
        for item1 in inp1_splt:     # TODO change 2-gram, measure effectiveness of 3-gram etc
            for item2 in inp2_splt:
                output = similarity(item1, item2)
                if output > max_output: 
                    max_output = output
                    print(item1, item2, max_output)


        inp1_splt = input1.split()
        inp2_splt = input2.split()     
        
        for item1 in inp1_splt:     # TODO change 2-gram, measure effectiveness of 3-gram etc
            for item2 in inp2_splt:
                output = similarity(item1, item2)
                if output > max_output: 
                    max_output = output
                    print(item1, item2, max_output)
                                   
        return str(max_output)

class Vector(Resource):
    def get(self, input):
        output = vector(input)
        return output


api.add_resource(MostSimilar, '/most_similar/<input>')
api.add_resource(Similarity, '/similarity/<input>')
api.add_resource(Vector, '/vector/<input>')


if __name__ == '__main__':
     app.run(host='0.0.0.0')
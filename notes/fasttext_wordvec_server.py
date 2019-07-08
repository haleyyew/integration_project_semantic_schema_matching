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


from gensim.test.utils import datapath
cap_path = datapath("/home/ubuntu/cc.en.300.bin")
from gensim.models.wrappers import FastText
wv = FastText.load_fasttext_format(cap_path)

CORS(app)

print('app started')

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
        output = similarity(input1, input2)
        return str(output)

class Vector(Resource):
    def get(self, input):
        output = vector(input)
        return output


api.add_resource(MostSimilar, '/most_similar/<input>')
api.add_resource(Similarity, '/similarity/<input>')
api.add_resource(Vector, '/vector/<input>')


if __name__ == '__main__':
     app.run(host='0.0.0.0')
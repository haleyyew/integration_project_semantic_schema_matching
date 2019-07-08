# ssh -i "/Users/haoran/Documents/thesis_schema_integration/notes/word2vec.pem" ubuntu@ec2-34-219-154-65.us-west-2.compute.amazonaws.com

from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

CORS(app)


def most_similar(word):
	return 1
     
def similarity(word1, word2):
    return 0
    
from ast import literal_eval
import numpy as np
def vector(word):
    a = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
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
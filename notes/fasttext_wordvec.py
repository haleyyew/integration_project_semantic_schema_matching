import http.client
import json

# connection = http.client.HTTPSConnection('127.0.0.1:5002')

# headers = {'Content-type': 'application/json'}

# foo = {'text': 'Hello world github/linguist#1 **cool**, and #1!'}
# json_foo = json.dumps(foo)

foo = 'Hello'
foo2 = 'park park__parking'

# connection.request('GET', '/most_similar/'+ foo)

import requests
server_ip = '52.151.20.94'

response = requests.get("http://"+server_ip+":5000/vector/"+foo)

import numpy as np
from ast import literal_eval
b = np.array(literal_eval(response.json()))
print(b)


response = requests.get("http://"+server_ip+":5000/most_similar/"+foo)
print(response.json())


response = requests.get("http://"+server_ip+":5000/similarity/"+foo2)
print(response.json(), float(response.json()))


# from __future__ import print_function
# from gensim.models import KeyedVectors
# import gensim
# from gensim.test.utils import datapath

# # cap_path = datapath("wordvec/wiki-news-300d-1M.vec")
# # fb_model = gensim.models.fasttext.load_facebook_model(cap_path)


# # Creating the model
# ## Takes a lot of time depending on the vector file size 
# en_model = KeyedVectors.load_word2vec_format('wordvec/wiki-news-300d-1M.vec')

# # Getting the tokens 
# words = []
# for word in en_model.vocab:
#     words.append(word)

# # Printing out number of tokens available
# print("Number of Tokens: {}".format(len(words)))

# print('computer', 'computer' in fb_model.wv.vocab)

# exit(0)

# # Printing out the dimension of a word vector 
# print("Dimension of a word vector: {}".format(
#     len(en_model[words[0]])
# ))

# # Print out the vector of a word 
# print("Vector components of a word: {}".format(
#     en_model[words[0]]
# ))

# # Pick a word 
# find_similar_to = 'car'

# # Finding out similar words [default= top 10]
# for similar_word in en_model.similar_by_word(find_similar_to):
#     print("Word: {0}, Similarity: {1:.2f}".format(
#         similar_word[0], similar_word[1]
#     ))

# # Output 
# # Word: cars, Similarity: 0.83
# # Word: automobile, Similarity: 0.72
# # Word: truck, Similarity: 0.71
# # Word: motorcar, Similarity: 0.70
# # Word: vehicle, Similarity: 0.70
# # Word: driver, Similarity: 0.69
# # Word: drivecar, Similarity: 0.69
# # Word: minivan, Similarity: 0.67
# # Word: roadster, Similarity: 0.67
# # Word: racecars, Similarity: 0.67

# # Test words 
# word_add = ['dhaka', 'india']
# word_sub = ['bangladesh']

# # Word vector addition and subtraction 
# for resultant_word in en_model.most_similar(
#     positive=word_add, negative=word_sub
# ):
#     print("Word : {0} , Similarity: {1:.2f}".format(
#         resultant_word[0], resultant_word[1]
#     ))

# # Output 

# # Word : delhi , Similarity: 0.77
# # Word : indore , Similarity: 0.76
# # Word : bangalore , Similarity: 0.75
# # Word : mumbai , Similarity: 0.75
# # Word : kolkata , Similarity: 0.75
# # Word : calcutta,india , Similarity: 0.75
# # Word : ahmedabad , Similarity: 0.75
# # Word : pune , Similarity: 0.74
# # Word : kolkata,india , Similarity: 0.74
# # Word : kolkatta , Similarity: 0.74
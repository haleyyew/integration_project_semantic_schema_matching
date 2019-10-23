import sys
path_lib = '/Users/haoran/Documents/neuralnet_hashing_and_similarity/tutorial_hybrid_flexmatcher/'
sys.path.insert(0, path_lib)
from flexmatcher import FlexMatcher

import pandas as pd

vals1 = [['year', 'Movie', 'imdb_rating'],
         ['2001', 'Lord of the Rings', '8.8'],
         ['2010', 'Inception', '8.7'],
         ['1999', 'The Matrix', '8.7']]
header = vals1.pop(0)
data1 = pd.DataFrame(vals1, columns=header)
data1_mapping = {'year': 'movie_year', 'imdb_rating': 'movie_rating',
                 'Movie': 'movie_name'}

# vals2 = [['title', 'produced', 'popularity'],
#          ['The Godfather', '1972', '9.2'],
#          ['Silver Linings Playbook', '2012', '7.8'],
#          ['The Big Short', '2015', '7.8']]
vals2 = [['title', 'popularity'],
         ['The Godfather',  '9.2'],
         ['Silver Linings Playbook',  '7.8'],
         ['The Big Short',  '7.8']]         
header = vals2.pop(0)
data2 = pd.DataFrame(vals2, columns=header)
# data2_mapping = {'popularity': 'movie_rating', 'produced': 'movie_year',
                 # 'title': 'movie_name'}
data2_mapping = {'popularity': 'movie_rating', 
                 'title': 'movie_name'}


vals3 = [['title', 'produced' ],
         ['The Godfather', '1972'],
         ['Silver Linings Playbook', '2012'],
         ['The Big Short', '2015']]
header = vals3.pop(0)
data3 = pd.DataFrame(vals3, columns=header)
data3_mapping = {'title': 'movie_name', 'produced': 'movie_year'}


schema_list = [data1, data2, data3]
mapping_list = [data1_mapping, data2_mapping, data3_mapping]

fm = FlexMatcher(schema_list, mapping_list, sample_size=100)
fm.train()                                           # train flexmatcher

# vals3 = [['rt', 'id', 'yr'],
#          ['8.5', 'The Pianist', '2002'],
#          ['7.7', 'The Social Network', '2010']]
vals3 = [['rt', 'id'],
         ['8.5', 'The Pianist'],
         ['7.7', 'The Social Network']]         
header = vals3.pop(0)
data3 = pd.DataFrame(vals3, columns=header)

predicted_mapping = fm.make_prediction(data3)

print ('FlexMatcher predicted that "rt" should be mapped to ' +
       predicted_mapping['rt'])
# print ('FlexMatcher predicted that "yr" should be mapped to ' +
#        predicted_mapping['yr'])
print ('FlexMatcher predicted that "id" should be mapped to ' +
       predicted_mapping['id'])

vals3 = [['rt', 'id', 'yr', '??'],
         ['8.5', 'The Pianist', '2002', '??'],
         ['7.7', 'The Social Network', '2010', '??']]       
header = vals3.pop(0)
data3 = pd.DataFrame(vals3, columns=header)

predicted_mapping = fm.make_prediction(data3)

print ('FlexMatcher predicted that "rt" should be mapped to ' +
       predicted_mapping['rt'])
print ('FlexMatcher predicted that "yr" should be mapped to ' +
       predicted_mapping['yr'])
print ('FlexMatcher predicted that "id" should be mapped to ' +
       predicted_mapping['id'])
print ('FlexMatcher predicted that "??" should be mapped to ' +
       predicted_mapping['??'])
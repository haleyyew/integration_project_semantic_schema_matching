import toy.data as td
import build_graphical_model as bgm
import numpy as np

print(td.KNOWLEDGE_BASE['park'])

# print(bgm.word2vec('this is a boss'))

from similarity.ngram import NGram
twogram = NGram(2)
print(twogram.distance('ABCD', 'ABTUIO'))


s1 = 'Adobe CreativeSuite 5 Master Collection from cheap 4zp'
s2 = 'Adobe CreativeSuite 5 Master Collection from cheap d1x'
fourgram = NGram(4)
print(fourgram.distance(s1, s2))
# print(twogram.distance(s1, s2))

# s2 = 'Adobe CreativeSuite 5 Master Collection from cheap 4zp'
# print(fourgram.distance(s1, s2))
#
# print(fourgram.distance('ABCD', 'ABTUIO'))

print(1-fourgram.distance(s1, s2))
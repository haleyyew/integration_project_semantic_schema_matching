import nltk
nltk.data.path.append('/Users/haoran/Documents/nltk_data/')

def wordnet():
    # nltk.download("wordnet", "/Users/haoran/Documents/nltk_data/")
    from nltk.corpus import wordnet

    syns = wordnet.synsets("program")
    print(syns[0].name())
    print(syns[0].lemmas()[0].name())

    print(syns[0].definition())
    print(syns[0].examples())

    synonyms = []
    antonyms = []

    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                for ant in l.antonyms():
                    antonyms.append(ant.name())

    for syn in wordnet.synsets("bad"):
        for l in syn.lemmas():
            antonyms.append(l.name())
            if l.antonyms():
                for ant in l.antonyms():
                    synonyms.append(ant.name())

    print(set(synonyms))
    print(set(antonyms))

    w1 = wordnet.synset('ship.n.01')
    w2 = wordnet.synset('boat.n.01')
    print(w1.wup_similarity(w2))

def naive_bayes():
    # nltk.download("movie_reviews", "/Users/haoran/Documents/nltk_data/")
    import random
    from nltk.corpus import movie_reviews

    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # print(documents[1])

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    # print(all_words.most_common(15))
    # print(all_words["stupid"])

    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
            # print(w, features[w])

        return features

    # print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    # count = 0
    # for item in featuresets:
    #     print(item)
    #     count += 1
    #     if count > 10:
    #         break

    # set that we'll train our classifier with
    training_set = featuresets[:1900]

    # set that we'll test against.
    testing_set = featuresets[1900:]

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    classifier.show_most_informative_features(15)

def scikit_learn():
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB

def hierarchical_clustering():
    import numpy as np
    import scipy.cluster.hierarchy as hac
    import matplotlib.pyplot as plt

    a = np.array([[0.1, 2.5],
                  [1.5, .4],
                  [0.3, 1],
                  [1, .8],
                  [0.5, 0],
                  [0, 0.5],
                  [0.5, 0.5],
                  [2.7, 2],
                  [2.2, 3.1],
                  [3, 2],
                  [3.2, 1.3]])

    fig, axes23 = plt.subplots(2, 3)

    for method, axes in zip(['single', 'complete'], axes23):
        z = hac.linkage(a, method=method)

        # Plotting
        axes[0].plot(range(1, len(z) + 1), z[::-1, 2])
        knee = np.diff(z[::-1, 2], 2)
        axes[0].plot(range(2, len(z)), knee)

        num_clust1 = knee.argmax() + 2
        knee[knee.argmax()] = 0
        num_clust2 = knee.argmax() + 2

        axes[0].text(num_clust1, z[::-1, 2][num_clust1 - 1], 'possible\n<- knee point')

        part1 = hac.fcluster(z, num_clust1, 'maxclust')
        part2 = hac.fcluster(z, num_clust2, 'maxclust')

        clr = ['#2200CC', '#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
               '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC']

        for part, ax in zip([part1, part2], axes[1:]):
            for cluster in set(part):
                ax.scatter(a[part == cluster, 0], a[part == cluster, 1],
                           color=clr[cluster])

        m = '\n(method: {})'.format(method)
        plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
                 ylabel='{}\ncluster distance'.format(m))
        plt.setp(axes[1], title='{} Clusters'.format(num_clust1))
        plt.setp(axes[2], title='{} Clusters'.format(num_clust2))

    plt.tight_layout()
    plt.show()
    return

def test_cosine_similarity():
    import re, math
    from collections import Counter

    WORD = re.compile(r'\w+')

    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)

    text1 = 'This is a foo bar sentence .'
    text2 = 'This sentence is similar to a foo bar sentence .'

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)
    print(cosine)


# ngram, wordnet, word2vec/gensim
hierarchical_clustering()

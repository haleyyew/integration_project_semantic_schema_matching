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

scikit_learn()
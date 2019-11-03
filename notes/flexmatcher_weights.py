
"""
Implement FlexMatcher.

This module is the main module of the FlexMatcher package and implements the
FlexMatcher class.

Todo:
    * Extend the module to work with and without data or column names.
    * Allow users to add/remove classifiers.
    * Combine modules (i.e., create_training_data and training functions).
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import flexmatcher.utils as utils
from sklearn import linear_model
import numpy as np
import pandas as pd
import pickle
import time

import re

# TODO folds=5 changed to 2, which is min of column len of all sets<== 
# TODO add more matchers

def columnAnalyzer(text):
    features = []
    words = re.findall('([a-z][a-z1-9]*|[1-9]+|[A-Z](?:[a-z1-9]+|[A-Z1-9]+))', text)
    for word in words:
        features.append(word.lower())
    return list(features)


class FlexMatcher:

    """Match a given schema to the mediated schema.

    The FlexMatcher learns to match an input schema to a mediated schema.
    The class considers panda dataframes as databases and their column names as
    the schema. FlexMatcher learn to do schema matching by training on
    instances of dataframes and how their columns are matched against the
    mediated schema.

    Attributes:
        train_data (dataframe): Dataframe with 3 columns. The name of
            the column in the schema, the value under that column and the name
            of the column in the mediated schema it was mapped to.
        col_train_data (dataframe): Dataframe  with 2 columns. The name
            the column in the schema and the name of the column in the mediated
            schema it was mapped to.
        data_src_num (int): Store the number of available data sources.
        classifier_list (list): List of classifiers used in the training.
        classifier_type (string): List containing the type of each classifier.
            Possible values are 'column' and 'value' classifiers.
        prediction_list (list): List of predictions on the training data
            produced by each classifier.
        weights (ndarray): A matrix where cell (i,j) captures how good the j-th
            classifier is at predicting if a column should match the i-th
            column (where columns are sorted by name) in the mediated schema.
        columns (list): The sorted list of column names in the mediated schema.
    """

    def __init__(self, dataframes, mappings, sample_size=300):
        """Prepares the list of classifiers that are being used for matching
        the schemas and creates the training data from the input datafames
        and their mappings.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        print('Create training data ...')
        self.create_training_data(dataframes, mappings, sample_size)
        print('Training data done ...')
        self.classifier_type = []

        # unigram_count_clf = NGramClassifier(ngram_range=(1, 1))
        # bigram_count_clf = NGramClassifier(ngram_range=(2, 2))
        # unichar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                         ngram_range=(1, 1))
        # bichar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                        ngram_range=(2, 2))
        # trichar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                         ngram_range=(3, 3))
        # quadchar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                          ngram_range=(4, 4))
        # char_dist_clf = CharDistClassifier()
        # self.classifier_list = [unigram_count_clf, bigram_count_clf,
        #                         unichar_count_clf, bichar_count_clf,
        #                         trichar_count_clf, quadchar_count_clf,
        #                         char_dist_clf]
        # self.classifier_type = ['value', 'value', 'value', 'value',
        #                         'value', 'value', 'value']
        # training sources
        # if self.data_src_num > 5:
        # if self.data_src_num > 2:
        #     col_char_dist_clf = CharDistClassifier()
        #     col_trichar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                                 ngram_range=(3, 3))
        #     col_quadchar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                                  ngram_range=(4, 4))
        #     col_quintchar_count_clf = NGramClassifier(analyzer='char_wb',
        #                                                   ngram_range=(5, 5))
        #     col_word_count_clf = \
        #         NGramClassifier(analyzer=columnAnalyzer)
        #     knn_clf = \
        #         KNNClassifier()
        #     self.classifier_list = self.classifier_list + \
        #         [col_char_dist_clf, col_trichar_count_clf,
        #          col_quadchar_count_clf, col_quintchar_count_clf,
        #          col_word_count_clf, knn_clf]
        #     self.classifier_type = self.classifier_type + (['column'] * 6)

        col_trichar_count_clf = NGramClassifier(analyzer='char_wb',
                                                        ngram_range=(3, 3))
        col_quadchar_count_clf = NGramClassifier(analyzer='char_wb',
                                                         ngram_range=(4, 4))
        self.classifier_list = [col_trichar_count_clf, col_quadchar_count_clf]
        self.classifier_type = ['column'] * 2

        # ngram_general_c = NGramGeneralClassifier()
        # wordnet_c = WordNetClassifier()
        # fasttext_c = FastTextClassifier()
        # self.classifier_list = [ngram_general_c, wordnet_c, fasttext_c]
        # self.classifier_type = ['column'] * 3


    def create_training_data(self, dataframes, mappings, sample_size):
        """Transform dataframes and mappings into training data.

        The method uses the names of columns as well as the data under each
        column as its training data. It also replaces missing values with 'NA'.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        train_data_list = []
        col_train_data_list = []
        for (datafr, mapping) in zip(dataframes, mappings):
            sampled_rows = datafr.sample(min(sample_size, datafr.shape[0]))
            sampled_data = pd.melt(sampled_rows)
            sampled_data.columns = ['name', 'value']
            # print(mapping)
            # print('==',sampled_data)
            sampled_data['class'] = \
                sampled_data.apply(lambda row: mapping[row['name']], axis=1)
            train_data_list.append(sampled_data)
            col_data = pd.DataFrame(datafr.columns)
            # print('=====',col_data)

            col_data.columns = ['name']
            col_data['value'] = col_data['name']
            col_data['class'] = \
                col_data.apply(lambda row: mapping[row['name']], axis=1)
            col_train_data_list.append(col_data)
        train_data = pd.concat(train_data_list, ignore_index=True)
        # print(train_data)
        self.train_data = train_data.fillna('NA')
        self.col_train_data = pd.concat(col_train_data_list, ignore_index=True)
        self.col_train_data = \
            self.col_train_data.drop_duplicates().reset_index(drop=True)
        self.data_src_num = len(dataframes)
        self.columns = \
            sorted(list(set.union(*[set(x.values()) for x in mappings])))
        # removing columns that are not present in the dataframe
        # TODO: this should change (It's not ideal to change problem definition
        # without notifying the user)
        available_columns = []
        for (datafr, mapping) in zip(dataframes, mappings):
                for c in datafr.columns:
                    available_columns.append(mapping[c])
        self.columns = sorted(list(set(available_columns)))

        print('=create_training_data= self.columns :')
        print(self.columns)

        print('=create_training_data= self.train_data :')
        print(self.train_data)

        print('=create_training_data= self.col_train_data :')
        print(self.col_train_data)

    def train(self):
        """Train each classifier and the meta-classifier."""
        self.prediction_list = []
        for (clf_inst, clf_type) in zip(self.classifier_list,
                                        self.classifier_type):
            start = time.time()
            # fitting the models and predict for training data
            if clf_type == 'value':
                clf_inst.fit(self.train_data)
                # predicting the training data
                self.prediction_list.append(clf_inst.predict_training())
                # print(clf_inst.predict_training())
            elif clf_type == 'column':
                clf_inst.fit(self.col_train_data)
                # predicting the training data
                col_data_prediction = \
                    pd.concat([pd.DataFrame(clf_inst.predict_training()),
                               self.col_train_data], axis=1)
                data_prediction = self.train_data.merge(col_data_prediction,
                                                        on=['name', 'class'],
                                                        how='left')
                data_prediction = np.asarray(data_prediction)
                data_prediction = \
                    data_prediction[:, range(3, 3 + len(self.columns))]
                self.prediction_list.append(data_prediction)
                
                print('=train= col_data_prediction: ')
                print(col_data_prediction)

                print('=train= data_prediction: ')
                print(len(data_prediction), len(data_prediction[0]))
                print(data_prediction)

            print('Train', time.time() - start)

        start = time.time()
        self.train_meta_learner()
        print('Train Meta: ' + str(time.time() - start))

    def train_meta_learner(self):
        """Train the meta-classifier.

        The data used for training the meta-classifier is the probability of
        assigning each point to each column (or class) by each classifier. The
        learned weights suggest how good each classifier is at predicting a
        particular class."""
        # suppressing a warning from scipy that gelsd is broken and gless is
        # being used instead.
        # warnings.filterwarnings(action="ignore", module="scipy",
        #                        message="^internal gelsd")
        coeff_list = []
        for class_ind, class_name in enumerate(self.columns):
            # preparing the dataset for logistic regression
            regression_data = self.train_data[['class']].copy()
            # print(regression_data)

            regression_data['is_class'] = \
                np.where(self.train_data['class'] == class_name, True, False)
            # adding the prediction probability from classifiers
            for classifier_ind, prediction in enumerate(self.prediction_list):
                regression_data['classifer' + str(classifier_ind)] = \
                    prediction[:, class_ind]

            # print(regression_data)

            # setting up the logistic regression
            stacker = linear_model.LogisticRegression(fit_intercept=True,
                                                      class_weight='balanced')
            stacker.fit(regression_data.iloc[:, 2:],
                        regression_data['is_class'])
            coeff_list.append(stacker.coef_.reshape(1, -1))

            print('=train_meta_learner= coeff_list :', class_ind, class_name)
            print(stacker.coef_.reshape(1, -1))

        self.weights = np.concatenate(tuple(coeff_list))
        print('=train_meta_learner= self.weights:')
        from pprint import pprint
        print('=====WEIGHTS=====')
        pprint(self.weights)
        print(np.sum(self.weights, axis = 0) )
        # print(np.sum(self.weights, axis = 1) )

    def make_prediction(self, data):
        """Map the schema of a given dataframe to the column of mediated schema.

        The procedure runs each classifier and then uses the weights (learned
        by the meta-trainer) to combine the prediction of each classifier.
        """
        data = data.fillna('NA').copy(deep=True)
        if data.shape[0] > 100:
            data = data.sample(100)
        # predicting each column
        predicted_mapping = {}
        for column in list(data):
            # print(column)
            column_dat = data[[column]]
            # print(column_dat)
            column_dat.columns = ['value']
            # print(column_dat.columns)
            column_name = pd.DataFrame({'value': [column]*column_dat.shape[0]})
            # print(column_name)
            scores = np.zeros((len(column_dat), len(self.columns)))
            # print(scores)
            for clf_ind, clf_inst in enumerate(self.classifier_list):
                if self.classifier_type[clf_ind] == 'value':
                    raw_prediction = clf_inst.predict(column_dat)
                elif self.classifier_type[clf_ind] == 'column':
                    raw_prediction = clf_inst.predict(column_name)
                # applying the weights to each class in the raw prediction
                for class_ind in range(len(self.columns)):
                    raw_prediction[:, class_ind] = \
                        (raw_prediction[:, class_ind] *
                         self.weights[class_ind, clf_ind])
                scores = scores + raw_prediction
            # print(scores)
            flat_scores = scores.sum(axis=0) / len(column_dat)
            # print(flat_scores)
            max_ind = flat_scores.argmax()
            predicted_mapping[column] = self.columns[max_ind]
        return predicted_mapping

    def save_model(self, name):
        """Serializes the FlexMatcher object into a model file using python's
        picke library."""
        with open(name + '.model', 'wb') as f:
            pickle.dump(self, f)



"""
Implement classifier for FlexMatcher.

This module defines an interface for classifiers.

Todo:
    * Implement more relevant classifiers.
    * Implement simple rules (e.g., does data match a phone number?).
    * Shuffle data before k-fold cutting in predict_training.
"""

from abc import ABCMeta, abstractmethod


class Classifier(object):

    """Define classifier interface for FlexMatcher."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, data):
        """Initialize the class."""
        pass

    @abstractmethod
    def fit(self, data):
        """Train based on the input training data."""
        pass

    @abstractmethod
    def predict_training(self, folds):
        """Predict the training data (using k-fold cross validation)."""
        pass

    @abstractmethod
    def predict(self, data):
        """Predict for unseen data."""
        pass


class NGramGeneralClassifier(Classifier):
    def __init__(self):
        return
    def fit(self, data):
        return
    def predict_training(self, folds=2):
        return
    def predict_proba_ordered(self, probs, classes):
        return
    def predict(self, data):
        return


class WordNetClassifier(Classifier):
    def __init__(self):
        return
    def fit(self, data):
        return
    def predict_training(self, folds=2):
        return
    def predict_proba_ordered(self, probs, classes):
        return
    def predict(self, data):
        return

class FastTextClassifier(Classifier):
    def __init__(self):
        return
    def fit(self, data):
        return
    def predict_training(self, folds=2):
        return
    def predict_proba_ordered(self, probs, classes):
        return
    def predict(self, data):
        return

from sklearn.model_selection import StratifiedKFold
# from flexmatcher.classify import Classifier
from sklearn import linear_model
import numpy as np


# DO NOT USE
class CharDistClassifier(Classifier):

    """Classify the data-point using counts of character types in the data.

    The CharDistClassifier extracts 7 simple features: number of
    white-space, digit, and alphabetical characters as well as their percentage
    and the total number of characters. Then it trains a logistic regression on
    top of these features.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        clf (LogisticRegression): The classifier instance.
        num_classes (int): Number of classes/columns to match to
        all_classes (ndarray): Sorted array of all possible classes
    """

    def __init__(self):
        """Initializes the classifier."""
        self.clf = linear_model.LogisticRegression(class_weight='balanced')

    def fit(self, data):
        """Extracts features and labels from the data and fits a model.

        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        self.all_classes = np.sort(np.unique(self.labels))
        # populating the features dataframe
        feat_df = data[['value']].copy()
        feat_df['length'] = feat_df['value'].apply(lambda val: len(val))
        feat_df['digit_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isdigit() for char in val) / len(val))
        feat_df['digit_num'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val))
        feat_df['alpha_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isalpha() for char in val) / len(val))
        feat_df['alpha_num'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val))
        feat_df['space_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isspace() for char in val) / len(val))
        feat_df['space_num'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val))
        self.features = feat_df.ix[:, 1:].as_matrix()
        # training the classifier
        self.clf.fit(self.features, self.labels)

    def predict_training(self, folds=2):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        print('-----------')
        print(self.features, self.labels)


        partial_clf = linear_model.LogisticRegression(class_weight='balanced')
        prediction = np.zeros((len(self.features), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)

        # try:
        #     for train_index, test_index in skf.split(self.features, self.labels):
        #         pass
        # except Exception:
        #     print('-------------')
        #     self.features = self.features + self.features
        #     self.labels = self.labels + self.labels

        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            curr_pred = partial_clf.predict_proba(test_features)
            prediction[test_index] = \
                self.predict_proba_ordered(curr_pred, partial_clf.classes_)
        return prediction

    def predict_proba_ordered(self, probs, classes):
        """Fills out the probability matrix with classes that were missing.

        Args:
            probs (list): list of probabilities, output of predict_proba
            classes_ (ndarray): list of classes from clf.classes_
            all_classes (ndarray): list of all possible classes
        """
        proba_ordered = np.zeros((probs.shape[0], self.all_classes.size),
                                 dtype=np.float)
        sorter = np.argsort(self.all_classes)
        idx = sorter[np.searchsorted(self.all_classes, classes, sorter=sorter)]
        proba_ordered[:, idx] = probs
        return proba_ordered

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        feat_df = data[['value']].copy()
        feat_df['length'] = feat_df['value'].apply(lambda val: len(val))
        feat_df['digit_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isdigit() for char in val) / len(val))
        feat_df['digit_num'] = feat_df['value'].apply(
            lambda val: sum(char.isdigit() for char in val))
        feat_df['alpha_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isalpha() for char in val) / len(val))
        feat_df['alpha_num'] = feat_df['value'].apply(
            lambda val: sum(char.isalpha() for char in val))
        feat_df['space_frac'] = feat_df['value'].apply(
            lambda val: 0 if len(val) == 0 else
            sum(char.isspace() for char in val) / len(val))
        feat_df['space_num'] = feat_df['value'].apply(
            lambda val: sum(char.isspace() for char in val))
        features = feat_df.ix[:, 1:].as_matrix()
        return self.clf.predict_proba(features)



from sklearn.model_selection import StratifiedKFold
# from flexmatcher.classify import Classifier
import numpy as np
import Levenshtein as lev

# NEED MODIFY
class KNNClassifier(Classifier):

    """Classify data-points (in string format) using their 3 nearest neighbors
    using the levenshtein distance metric.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        strings (list): List of strings for which the labels are provided.
        num_classes (int): Number of classes/columns to match to.
        column_index (dict): Dictionary mapping each column to its index.
    """

    def __init__(self):
        """Initializes the classifier."""
        pass

    def fit(self, data):
        """Store the strings and their corresponding labels.

        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.strings = np.array(data['value'])
        self.num_classes = len(data['class'].unique())
        self.column_index = dict(zip(sorted(list(data['class'].unique())),
                                     range(self.num_classes)))

    def predict_training(self, folds=2):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        prediction = np.zeros((len(self.strings), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.strings, self.labels):
            # prepare the training and test data
            training_strings = self.strings[train_index]
            test_strings = self.strings[test_index]
            training_labels = self.labels[train_index]
            # predicting the results
            part_prediction = self.find_knn(training_strings, training_labels,
                                            test_strings)
            prediction[test_index] = part_prediction
        return prediction

    def find_knn(self, train_strings, train_labels, test_strings):
        """Find 3 nearest neighbors of each item in test_strings in
        train_strings and report their labels as the prediction.

        Args:
            train_strings (ndarray): Numpy array with strings in training set
            train_labels (ndarray): Numpy array with labels of train_strings
            test_strings (ndarray): Numpy array with string to be predict for
        """
        prediction = np.zeros((len(test_strings), self.num_classes))
        for i in range(len(test_strings)):
            a_str = test_strings[i]
            dists = np.array([0] * len(train_strings))
            for j in range(len(train_strings)):
                b_str = train_strings[j]
                dists[j] = lev.distance(a_str, b_str)
            # finding the top 3
            top3 = dists.argsort()[:3]
            for ind in top3:
                prediction[i][self.column_index[train_labels[ind]]] += 1.0 / 3
        return prediction

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        input_strings = np.array(data['value'])
        return self.find_knn(self.strings, self.labels, input_strings)



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
# from flexmatcher.classify import Classifier
import numpy as np


# NEED MODIFY
class NGramClassifier(Classifier):

    """Classify data-points using counts of n-gram sequence of words or chars.

    The NGramClassifier uses n-grams of words or characters (based on user
    preference) and extracts count features or binary features (based on user
    preference) to train a classifier. It uses a LogisticRegression
    classifier as its training model.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        vectorizer (object): Vectorizer for transforming text to features. It
        will be either of type CountVectorizer or HashingVectorizer.
        clf (LogisticRegression): The classifier instance.
        num_classes (int): Number of classes/columns to match to
        all_classes (ndarray): Sorted array of all possible classes
    """

    def __init__(self, ngram_range=(1, 1), analyzer='word', count=True,
                 n_features=200):
        """Initializes the classifier.

        Args:
            ngram_range (tuple): Pair of ints specifying the range of ngrams.
            analyzer (string): Determines what type of analyzer to be used.
            Setting it to 'word' will consider each word as a unit of language
            and 'char' will consider each character as a unit of language.
            count (boolean): Determines if features are counts of n-grams
            versus a binary value encoding if the n-gram is present or not.
            n_features (int): Maximum number of features used.
        """
        # checking what type of vectorizer to create
        if count:
            self.vectorizer = CountVectorizer(analyzer=analyzer,
                                              ngram_range=ngram_range,
                                              max_features=n_features)
        else:
            self.vectorizer = HashingVectorizer(analyzer=analyzer,
                                                ngram_range=ngram_range,
                                                n_features=n_features)

    def fit(self, data):
        """
        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        self.all_classes = np.sort(np.unique(self.labels))
        values = list(data['value'])
        self.features = self.vectorizer.fit_transform(values).toarray()
        # training the classifier
        self.lrm = linear_model.LogisticRegression(class_weight='balanced')
        self.lrm.fit(self.features, self.labels)

        print('=NGramClassifier.fit= self.all_classes:')
        print(self.all_classes)

        print('=NGramClassifier.fit= self.features:')
        print(self.features)


    def predict_training(self, folds=2):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        partial_clf = linear_model.LogisticRegression(class_weight='balanced')
        prediction = np.zeros((len(self.features), self.num_classes))

        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            curr_pred = partial_clf.predict_proba(test_features)
            prediction[test_index] = \
                self.predict_proba_ordered(curr_pred, partial_clf.classes_)
            # print('test_index', test_index)

        print('=NGramClassifier.predict_training= prediction :')
        print(prediction)
            
        return prediction

    def predict_proba_ordered(self, probs, classes):
        """Fills out the probability matrix with classes that were missing.

        Args:
            probs (list): list of probabilities, output of predict_proba
            classes_ (ndarray): list of classes from clf.classes_
            all_classes (ndarray): list of all possible classes
        """
        proba_ordered = np.zeros((probs.shape[0], self.all_classes.size),
                                 dtype=np.float)
        sorter = np.argsort(self.all_classes)
        # print('sorter', sorter)
        idx = sorter[np.searchsorted(self.all_classes, classes, sorter=sorter)]
        # print('idx', idx)
        proba_ordered[:, idx] = probs
        # print('probs')
        # print(probs)

        print('=NGramClassifier.predict_proba_ordered= proba_ordered[:, idx]')
        print(proba_ordered[:, idx])

        print('=NGramClassifier.predict_proba_ordered= proba_ordered')
        print(proba_ordered)
        return proba_ordered

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        values = list(data['value'])
        features = self.vectorizer.transform(values).toarray()
        return self.lrm.predict_proba(features)

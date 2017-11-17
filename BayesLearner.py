import numpy as np
from collections import defaultdict, Counter, OrderedDict
import math
import copy
import itertools
import operator


class NaiveBayesLearner(object):
    def __init__(self, meta):
        self.meta = meta
        self.features = meta.names()
        self.features_list = OrderedDict()
        for feature in self.features:
            self.features_list[feature] = self.meta[feature][1]
        self.probabilities = OrderedDict()
        self.y_probabilities = OrderedDict()

    # Method that is called to train a naive bayes model.
    def train(self, data):
        self.set_probabilities(data)

    # Method that is called to test a dataset on the model.
    def test(self, data):
        correct_predictions = self.classify(data)
        return float(correct_predictions) / len(data)

    # Method that executes classification of test data on model.
    def classify(self, data):
        y_feat = self.features[-1]
        correct_predictions = 0
        for sample in data:
            curr_y_probabilities = self.y_probabilities.copy()
            index = 0
            # Get product of probabilities based off of feature values in current sample.
            for val in sample:
                if val != sample[-1]:
                    feat = self.features[index]
                    for y_val in self.features_list[y_feat]:
                        curr_y_probabilities[y_val] *= self.probabilities[feat][val][y_val]
                index += 1

            # Determine the cumulative probabilities to get future denominator.
            cum_y_probabilities = 0
            for y_val in self.features_list[y_feat]:
                cum_y_probabilities += curr_y_probabilities[y_val]

            # Determine final probabilities for different y values.
            final_y_probabilities = OrderedDict()
            for y_val in self.features_list[y_feat]:
                final_y_probabilities[y_val] = float(float(curr_y_probabilities[y_val]) / float(cum_y_probabilities))

            # Determine the most likely final probability to perform classification.
            highest_y_probability = 0
            highest_y_val = None
            for y_val in self.features_list[y_feat]:
                if final_y_probabilities[y_val] > highest_y_probability:
                    highest_y_probability = final_y_probabilities[y_val]
                    highest_y_val = y_val

            print highest_y_val + " " + sample[-1] + " ",
            print ("%.12f" % highest_y_probability)
            if highest_y_val == sample[-1]:
                correct_predictions += 1

        print ""
        print correct_predictions
        return correct_predictions

    # Set the probabilities field.
    def set_probabilities(self, data):
        size = len(data)

        # Determine counts of all y values.
        y_counts = OrderedDict()
        y_feat = self.features[-1]
        for y_val in self.features_list[y_feat]:
            y_counts[y_val] = 0
            for sample in data:
                if sample[y_feat] == y_val:
                    y_counts[y_val] += 1
            self.y_probabilities[y_val] = float(y_counts[y_val] + 1)/(size + 2)

        # Determine counts of all values of features with respect to the y values.
        _counts = OrderedDict()
        for feat in self.features:
            _counts[feat] = OrderedDict()
            for val in self.features_list[feat]:
                _counts[feat][val] = OrderedDict()
                for y_val in self.features_list[y_feat]:
                    _counts[feat][val][y_val] = 0
                    for sample in data:
                        if sample[feat] == val and sample[y_feat] == y_val:
                            _counts[feat][val][y_val] += 1

        # Calculate the probabilities of feature values with respect to the y values.
        for feat in self.features:
            if feat != self.features[-1]:
                self.probabilities[feat] = OrderedDict()
                num_of_vals = len(self.features_list[feat])
                for val in self.features_list[feat]:
                    self.probabilities[feat][val] = OrderedDict()
                    for y_val in self.features_list[y_feat]:
                        self.probabilities[feat][val][y_val] = float(_counts[feat][val][y_val] + 1) / \
                                                               (y_counts[y_val] + num_of_vals)

    # Print out all features in the data.
    def print_features(self):
        for feat in self.features:
            if feat != self.features[-1]:
                print feat + " " + self.features[-1]
        print ""


class TAN(object):
    def __init__(self, meta):
        self.meta = meta
        self.features = meta.names()
        self.features_list = OrderedDict()
        for feature in self.features:
            self.features_list[feature] = self.meta[feature][1]
        self.probabilities_naive_bayes = OrderedDict()
        self.probabilities_tan = OrderedDict()
        self.probabilities_tan_cond = OrderedDict()
        self.probabilities_tan_final = OrderedDict()
        self.y_probabilities = OrderedDict()
        self.vertices = list()
        self.edges = list()
        self.parents = OrderedDict()
        self.w = OrderedDict()

    # Method that is called to train a tree augmented naive bayes model.
    def train(self, data):
        self.set_probabilities_naive_bayes(data)
        self.set_weights(data)
        self.perform_prim()
        self.set_probabilities(data)

    # Method that is called to test a data set on the tree augmented naive bayes model.
    def test(self, data):
        correct_predictions = self.classify(data)
        return float(correct_predictions) / len(data)

    # Handles classification of data set on current model.
    def classify(self, data):
        y_feat = self.features[-1]
        correct_predictions = 0
        for sample in data:
            curr_y_probabilities = self.y_probabilities.copy()
            index = 0
            # Calculate the product of the probabilities for the current sample by iteratively multiplying based off
            # of the features of the sample.
            for val in sample:
                if val != sample[-1]:
                    feat = self.features[index]
                    # If the feature is not the root.
                    if feat in self.parents:
                        parent_feat = self.parents[feat]
                        parent_val = sample[self.get_index(parent_feat)]
                        for y_val in self.features_list[y_feat]:
                            curr_y_probabilities[y_val] *= \
                                self.probabilities_tan_final[feat][parent_feat][val][parent_val][y_val]
                    # If the feature is the root.
                    else:
                        for y_val in self.features_list[y_feat]:
                            curr_y_probabilities[y_val] *= self.probabilities_naive_bayes[feat][val][y_val]
                index += 1

            # Determine the cumulative probabilities to get future denominator.
            cum_y_probabilities = 0
            for y_val in self.features_list[y_feat]:
                cum_y_probabilities += curr_y_probabilities[y_val]

            # Determine final probabilities for different y values.
            final_y_probabilities = OrderedDict()
            for y_val in self.features_list[y_feat]:
                final_y_probabilities[y_val] = float(float(curr_y_probabilities[y_val]) / float(cum_y_probabilities))

            # Determine the most likely final probability to perform classification.
            highest_y_probability = 0
            highest_y_val = None
            for y_val in self.features_list[y_feat]:
                if final_y_probabilities[y_val] > highest_y_probability:
                    highest_y_probability = final_y_probabilities[y_val]
                    highest_y_val = y_val

            print highest_y_val + " " + sample[-1] + " ",
            print ("%.12f" % highest_y_probability)
            if highest_y_val == sample[-1]:
                correct_predictions += 1

        print ""
        print correct_predictions
        return correct_predictions

    # Return the index of the feature
    def get_index(self, feat):
        for i in range(0, len(self.features) - 1):
            if self.features[i] == feat:
                return i
        return -1

    # Set the probabilities fields according to naive bayes.
    def set_probabilities_naive_bayes(self, data):
        size = len(data)

        # Determine counts of all y values.
        y_counts = OrderedDict()
        y_feat = self.features[-1]
        for y_val in self.features_list[y_feat]:
            y_counts[y_val] = 0
            for sample in data:
                if sample[y_feat] == y_val:
                    y_counts[y_val] += 1
            self.y_probabilities[y_val] = float(y_counts[y_val] + 1) / (size + 2)

        # Determine counts of all values of features with respect to the y values.
        _counts = OrderedDict()
        for feat in self.features:
            _counts[feat] = OrderedDict()
            for val in self.features_list[feat]:
                _counts[feat][val] = OrderedDict()
                for y_val in self.features_list[y_feat]:
                    _counts[feat][val][y_val] = 0
                    for sample in data:
                        if sample[feat] == val and sample[y_feat] == y_val:
                            _counts[feat][val][y_val] += 1

        # Calculate the probabilities of feature values with respect to the y values.
        for feat in self.features:
            if feat != self.features[-1]:
                self.probabilities_naive_bayes[feat] = OrderedDict()
                num_of_vals = len(self.features_list[feat])
                for val in self.features_list[feat]:
                    self.probabilities_naive_bayes[feat][val] = OrderedDict()
                    for y_val in self.features_list[y_feat]:
                        self.probabilities_naive_bayes[feat][val][y_val] = float(_counts[feat][val][y_val] + 1) / \
                                                               (y_counts[y_val] + num_of_vals)

    # Set weights in graph between nodes of all combinations of feature values.
    def set_weights(self, data):
        size = len(data)

        # Determine counts of y values.
        y_counts = OrderedDict()
        y_feat = self.features[-1]
        for y_val in self.features_list[y_feat]:
            y_counts[y_val] = 0
            for sample in data:
                if sample[y_feat] == y_val:
                    y_counts[y_val] += 1

        # Determine counts of occurences of (feat/val)/(feat/val) combos wrt y values.
        _counts = OrderedDict()
        for feat1 in self.features:
            _counts[feat1] = OrderedDict()
            if feat1 != self.features[-1]:
                for feat2 in self.features:
                    _counts[feat1][feat2] = OrderedDict()
                    if feat2 != self.features[-1] and feat2 != feat1:
                        for val1 in self.features_list[feat1]:
                            _counts[feat1][feat2][val1] = OrderedDict()
                            for val2 in self.features_list[feat2]:
                                _counts[feat1][feat2][val1][val2] = OrderedDict()
                                for y_val in self.features_list[y_feat]:
                                    _counts[feat1][feat2][val1][val2][y_val] = 0
                                    for sample in data:
                                        if sample[feat1] == val1 and sample[feat2] == val2 and sample[y_feat] == y_val:
                                            _counts[feat1][feat2][val1][val2][y_val] += 1

        # Calculate weights between features by summing up the probabilities of (feat/val)/(feat/val) combinations
        # over the y values.
        for feat1 in self.features:
            if feat1 != self.features[-1]:
                self.probabilities_tan[feat1] = OrderedDict()
                self.probabilities_tan_cond[feat1] = OrderedDict()
                self.w[feat1] = OrderedDict()
                for feat2 in self.features:
                    if feat2 != self.features[-1] and feat2 != feat1:
                        self.probabilities_tan[feat1][feat2] = OrderedDict()
                        self.probabilities_tan_cond[feat1][feat2] = OrderedDict()
                        num_vals1 = len(self.features_list[feat1])
                        num_vals2 = len(self.features_list[feat2])
                        #print("zeroed")
                        summ = 0
                        for val1 in self.features_list[feat1]:
                            self.probabilities_tan[feat1][feat2][val1] = OrderedDict()
                            self.probabilities_tan_cond[feat1][feat2][val1] = OrderedDict()
                            for val2 in self.features_list[feat2]:
                                self.probabilities_tan[feat1][feat2][val1][val2] = OrderedDict()
                                self.probabilities_tan_cond[feat1][feat2][val1][val2] = OrderedDict()
                                for y_val in self.features_list[y_feat]:
                                    self.probabilities_tan[feat1][feat2][val1][val2][y_val] = \
                                        float(_counts[feat1][feat2][val1][val2][y_val] + 1) / (size +
                                                                                               2*num_vals1 * num_vals2)
                                    self.probabilities_tan_cond[feat1][feat2][val1][val2][y_val] = \
                                        float(_counts[feat1][feat2][val1][val2][y_val] + 1) / (y_counts[y_val] +
                                                                                               num_vals1 * num_vals2)
                                    summ += self.probabilities_tan[feat1][feat2][val1][val2][y_val] \
                                           * math.log(self.probabilities_tan_cond[feat1][feat2][val1][val2][y_val] /
                                                      (self.probabilities_naive_bayes[feat1][val1][y_val] *
                                                       self.probabilities_naive_bayes[feat2][val2][y_val]), 2)
                                    #print(summ)
                            self.w[feat1][feat2] = summ

    # Perform Prim's Algorithm to find and construct the minimum spanning tree.
    def perform_prim(self):
        # Append first feature.
        self.vertices.append(self.features[0])
        # Iterate until all features are included (ignoring y_feat).
        while len(self.vertices) < len(self.features) - 1:
            # Get maximum weight edge.
            max_weight_val = 0
            edge_max = None
            for vertex in self.vertices:
                for feat in self.features:
                    if feat != self.features[-1] and feat not in self.vertices:
                        curr_weight = self.w[vertex][feat]
                        if curr_weight >= max_weight_val:
                            max_weight_val = curr_weight
                            edge_max = (curr_weight, vertex, feat)

            # Add new vertex and edge.
            self.vertices.append(edge_max[2])
            self.edges.append(edge_max)
            self.parents[edge_max[2]] = edge_max[1]
           #print(edge_max)

        for feat in self.features:
            if feat == self.features[-1]:
                print ""
            elif feat in self.parents:
                print feat + " " + self.parents[feat] + " " + self.features[-1]
            else:
                print feat + " " + self.features[-1]

    # Determine probabilities of
    def set_probabilities(self, data):
        y_feat = self.features[-1]
        # Get counts of parent (feat/val) happening wrt to y values and counts of parent/child (feat/val)/(feat/val)
        # occuring wrt to the y values.
        _counts_both = OrderedDict()
        _counts_parent = OrderedDict()
        for feat in self.features:
            _counts_both[feat] = OrderedDict()
            if feat != self.features[-1] and feat in self.parents:
                parent_feat = self.parents[feat]
                _counts_both[feat][parent_feat] = OrderedDict()
                _counts_parent[parent_feat] = OrderedDict()
                for val in self.features_list[feat]:
                    _counts_both[feat][parent_feat][val] = OrderedDict()
                    for parent_val in self.features_list[parent_feat]:
                        _counts_both[feat][parent_feat][val][parent_val] = OrderedDict()
                        _counts_parent[parent_feat][parent_val] = OrderedDict()
                        for y_val in self.features_list[y_feat]:
                            _counts_both[feat][parent_feat][val][parent_val][y_val] = 0
                            _counts_parent[parent_feat][parent_val][y_val] = 0
                            for sample in data:
                                if sample[parent_feat] == parent_val and sample[y_feat] == y_val:
                                    _counts_parent[parent_feat][parent_val][y_val] += 1
                                    if sample[feat] == val:
                                        _counts_both[feat][parent_feat][val][parent_val][y_val] += 1

        # Calculate the conditional probabilities based off of parent (feat/val) and child (feat/val) wrt to y values.
        for feat in self.features:
            if feat != self.features[-1] and feat in self.parents:
                feat_vals = len(self.features_list[feat])
                parent_feat = self.parents[feat]
                self.probabilities_tan_final[feat] = OrderedDict()
                self.probabilities_tan_final[feat][parent_feat] = OrderedDict()
                for val in self.features_list[feat]:
                    self.probabilities_tan_final[feat][parent_feat][val] = OrderedDict()
                    for parent_val in self.features_list[parent_feat]:
                        self.probabilities_tan_final[feat][parent_feat][val][parent_val] = OrderedDict()
                        for y_val in self.features_list[y_feat]:
                            self.probabilities_tan_final[feat][parent_feat][val][parent_val][y_val] = \
                                float(_counts_both[feat][parent_feat][val][parent_val][y_val] + 1) / \
                                (_counts_parent[parent_feat][parent_val][y_val] + feat_vals)

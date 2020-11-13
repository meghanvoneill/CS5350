import csv
import math
import random
from pprint import pprint


def main():

    # Build dictionary from data set.
    data_set = {'examples': [], 'attributes': set()}

    # [Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...]
    with open('data/a1a.train') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            new_row = row[0].split(' ')
            new_row = new_row[0:-1]
            new_example = {'label': int(new_row[0])}
            for data in new_row[1:]:
                arr = data.split(':')
                idx = arr[0]
                val = arr[1]
                new_example[idx] = int(val)
                data_set['attributes'].add(idx)

            data_set['examples'].append(new_example)

    test_set = {'examples': [], 'attributes': set()}

    # [Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...]
    with open('data/a1a.test') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            new_row = row[0].split(' ')
            new_row = new_row[0:-1]
            new_example = {'label': int(new_row[0])}
            for data in new_row[1:]:
                arr = data.split(':')
                idx = arr[0]
                val = arr[1]
                new_example[idx] = int(val)
                test_set['attributes'].add(idx)

            test_set['examples'].append(new_example)

    # Experiments:

    # Build Decision Tree using training files.
    # experiment_1(data_set, test_set)

    # Build Decision Tree using training files.
    # experiment_2(data_set, test_set)

    # Toy data set from Professor Vivek Srikumar's lectures on Decision Trees.
    toy_attributes = ['O', 'T', 'H', 'W']
    toy_data = {'examples': [{'label': -1, 'O': 'S', 'T': 'H', 'H': 'H', 'W': 'W'},
                             {'label': -1, 'O': 'S', 'T': 'H', 'H': 'H', 'W': 'S'},
                             {'label': +1, 'O': 'O', 'T': 'H', 'H': 'H', 'W': 'W'},
                             {'label': +1, 'O': 'R', 'T': 'M', 'H': 'H', 'W': 'W'},
                             {'label': +1, 'O': 'R', 'T': 'C', 'H': 'N', 'W': 'W'},
                             {'label': -1, 'O': 'R', 'T': 'C', 'H': 'N', 'W': 'S'},
                             {'label': +1, 'O': 'O', 'T': 'C', 'H': 'N', 'W': 'S'},
                             {'label': -1, 'O': 'S', 'T': 'M', 'H': 'H', 'W': 'W'},
                             {'label': +1, 'O': 'S', 'T': 'C', 'H': 'N', 'W': 'W'},
                             {'label': +1, 'O': 'R', 'T': 'M', 'H': 'N', 'W': 'W'},
                             {'label': +1, 'O': 'S', 'T': 'M', 'H': 'N', 'W': 'S'},
                             {'label': +1, 'O': 'O', 'T': 'M', 'H': 'H', 'W': 'S'},
                             {'label': +1, 'O': 'O', 'T': 'H', 'H': 'N', 'W': 'W'},
                             {'label': -1, 'O': 'R', 'T': 'M', 'H': 'H', 'W': 'S'}, ],
                'attributes': set(toy_attributes)}
    # tree, tree_depthlimit = experiment_3(toy_data)

    # Report:
    # (a) Most common label in the training data:
    print('(a) Most common label: ' + str(common_label(data_set)))
    # (b) Entropy of the training data:
    print('(b) Entropy of the training data: ' + str(entropy(data_set)))
    # (c) Best feature and its information gain in training data:
    best_feature = determine_best_attribute(data_set, data_set['attributes'])
    info_gain = information_gain(data_set, best_feature)
    print('(c) Best feature and its information gain: ' + str(best_feature) + ', ' + str(info_gain))
    # (d) Accuracy of the training set:
    training_accuracy, test_accuracy = experiment_1(data_set, test_set)
    print('(d) Training accuracy: ' + str(training_accuracy))
    # (e) Accuracy of the test set:
    print('(e) Test accuracy: ' + str(test_accuracy))
    # (f) Average accuracies from cross-validation by depth:
    test_results, averages = cross_validation(5, 'data/CVfolds/fold')
    print('(f) Average accuracies from cross-validation by depth:')
    print('\t' + str(averages))
    # (g) Best depth:
    print('(g) Best depth is: ')
    print('\t 3 or 4')
    # (h) Accuracy on the test set using the best depth:
    training_accuracy, test_accuracy = experiment_2(data_set, test_set)
    print('(h) Accuracy on the test set using best depth of 3: ' + str(test_accuracy))

    return


def experiment_1(data_set, test_set):

    # Build Decision Tree using training files.
    tree = build_decision_tree(data_set)
    train_accuracy = accuracy(tree, data_set['examples'])
    test_accuracy = accuracy(tree, test_set['examples'])
    return train_accuracy, test_accuracy


def experiment_2(data_set, test_set):

    # Build Decision Tree using training files and depth-limit.
    tree = build_decision_tree(data_set, 3)
    train_accuracy = accuracy(tree, data_set['examples'])
    test_accuracy = accuracy(tree, test_set['examples'])
    return train_accuracy, test_accuracy


def experiment_3(data_set):

    tree = build_decision_tree(data_set)
    tree_plus_depthlimit = build_decision_tree(data_set, 2)

    return tree, tree_plus_depthlimit


def traverse_tree(tree, example):

    current_node = tree['root']

    while len(current_node['branches'].keys()) > 0:
        attribute = current_node['value']
        if attribute in example.keys():
            value = example[attribute]
        else:
            value = 0
        current_node = current_node['branches'][value]

    return current_node['value']


def cross_validation(number_of_folds, file_name):

    folds = []

    for i in range(1, number_of_folds+1):
        fold_name = file_name + str(i)
        fold = {'examples': [], 'attributes': set()}

        # [Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...]
        with open(fold_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\n')
            for row in csv_reader:
                new_row = row[0].split(' ')
                new_row = new_row[0:-1]
                new_example = {'label': int(new_row[0])}
                for data in new_row[1:]:
                    arr = data.split(':')
                    idx = arr[0]
                    val = arr[1]
                    new_example[idx] = int(val)
                    fold['attributes'].add(idx)

                fold['examples'].append(new_example)
        folds.append(fold)

    train_results, test_results = cv_testing_with_variable_depths(folds)

    # Find averages.
    averages = {}
    for key in test_results.keys():
        summation = .0
        for acc in test_results[key]:
            summation += acc
        averages[key] = summation / len(test_results.keys())
        # print(str(key) + ': ' + str(summation) + ' / ' + str(len(test_results.keys())) + ' = ' + str(averages[key]))

    return test_results, averages


def cv_testing_with_variable_depths(folds):

    depth_train_results = {}
    depth_test_results = {}

    # For variable depths, [1,6).
    for depth in range(1, 6):
        depth_train_results[depth] = []
        depth_test_results[depth] = []

        # For each heldout fold:
        for test_fold in range(len(folds)):
            all_but_one = {'examples': [], 'attributes': set()}
            for train_fold in range(len(folds)):
                if train_fold != test_fold:
                    all_but_one['examples'] += folds[train_fold]['examples']
                    all_but_one['attributes'] = all_but_one['attributes'].union(folds[train_fold]['attributes'])

            tree = build_decision_tree(all_but_one, depth)
            train_accuracy = accuracy(tree, all_but_one['examples'])
            test_accuracy = accuracy(tree, folds[test_fold]['examples'])
            depth_train_results[depth].append(train_accuracy)
            depth_test_results[depth].append(test_accuracy)

    return depth_train_results, depth_test_results


def accuracy(tree, test_examples):

    correct = .0
    size = len(test_examples)

    for example in test_examples:
        test_label = traverse_tree(tree, example)
        label = example['label']
        if test_label == label:
            correct += 1

    return correct / size


def merge(dictionary_1, dictionary_2):

    new_dictionary = {}

    if len(dictionary_1['examples']) < 1 and len(dictionary_2['examples']) < 1:
        new_dictionary['examples'] = []
        new_dictionary['attributes'] = set()
        return new_dictionary
    elif len(dictionary_1['examples']) < 1:
        new_dictionary['examples'] = dictionary_2['examples']
        new_dictionary['attributes'] = dictionary_2['attributes']
    elif len(dictionary_2['examples']) < 1:
        new_dictionary['examples'] = dictionary_1['examples']
        new_dictionary['attributes'] = dictionary_1['attributes']
    else:
        new_dictionary['examples'] = dictionary_1['examples'] + dictionary_2['examples']
        new_dictionary['attributes'] = dictionary_1['attributes'].union(dictionary_2['attributes'])

    return new_dictionary


def build_decision_tree(data_set, depth_limit=None):

    if depth_limit:
        tree = {'depth': 0, 'root': ID3_depth_limit(data_set, data_set['attributes'], depth_limit, 0)}
    else:
        tree = {'depth': 0, 'root': ID3(data_set, data_set['attributes'], 0)}

    return tree


def ID3(tree, attributes, depth):

    labels = all_known_attribute_values(tree, 'label')

    if len(labels) == 1:
        return {'value': tree['examples'][0]['label'], 'branches': {}}

    att = determine_best_attribute(tree, attributes)
    root = {'value': att, 'branches': {}}
    values_of_att = all_known_attribute_values(tree, att)

    for v in values_of_att:
        # Add a new tree branch for attribute A taking value v:
        # Let S_v be the subset of examples in S with A=v.
        sub_set = subset(tree, att, v)

        # If S_v is empty:
        if len(sub_set) < 1:
            # Add leaf node with the common value of Label in S
            label = common_label(tree)
            root['branches'][v] = {'value': label, 'branches': {}}

        # Otherwise, below this branch, add the subtree ID3(S_v, Attributes - {A})
        else:
            root['branches'][v] = ID3(sub_set, sub_set['attributes'], depth+1)
    root['depth'] = depth
    return root


def ID3_depth_limit(tree, attributes, depth_limit, depth):
    labels = all_known_attribute_values(tree, 'label')

    if len(labels) == 1:
        return {'value': tree['examples'][0]['label'], 'branches': {}}

    att = determine_best_attribute(tree, attributes)
    root = {'value': att, 'branches': {}}
    values_of_att = all_known_attribute_values(tree, att)

    for v in values_of_att:
        # Add a new tree branch for attribute A taking value v.
        # Let S_v be the subset of examples in S with A=v.
        sub_set = subset(tree, att, v)

        # If S_v is empty:
        if len(sub_set['examples']) < 1:
            # Add leaf node with the common value of Label in S
            label = common_label(tree)
            root['branches'][v] = {'value': label, 'branches': {}}
        # Otherwise, below this branch, add the subtree ID3(S_v, Attributes - {A})
        else:
            if depth >= depth_limit - 1:
                # Add leaf node with the common value of Label in S
                label = common_label(sub_set)
                root['branches'][v] = {'value': label, 'branches': {}}
            else:
                root['branches'][v] = ID3_depth_limit(sub_set, sub_set['attributes'], depth_limit, depth+1)
    root['depth'] = depth

    return root


def determine_best_attribute(data_set, attributes):

    best_info_gain = 0
    best_gain_att = None

    for a in attributes:
        gain = information_gain(data_set, a)
        if gain > best_info_gain:
            best_info_gain = gain
            best_gain_att = a

    return best_gain_att


# The information gain of the given attribute is the expected reduction
#   in entropy caused by partitioning on this attribute.
def information_gain(data_set, attribute):
    gain = entropy(data_set)
    summation = 0

    values_of_att = [0,1]
    for value in values_of_att:
        subset_of_value = subset(data_set, attribute, value)
        if len(subset_of_value['examples']) > 0:
            entropy_of_subset = entropy(subset_of_value)
            summation += (len(subset_of_value['examples']) / len(data_set['examples'])) * entropy_of_subset

    return gain - summation


# Calculates the entropy of a given data set. The ratio of positive labels is defined as the number of labels greater
#   than 0, over the total number of labels, and the ratio of negative labels is the number of remaining labels, over
#   the total number of labels.
def entropy(data_set):

    positive_examples = 0
    negative_examples = 0
    total_examples = 0

    for example in data_set['examples']:
        if example['label'] > 0:
            positive_examples += 1
        else:
            negative_examples += 1
        total_examples += 1

    proportion_of_p = positive_examples / total_examples
    proportion_of_n = negative_examples / total_examples

    if positive_examples == 0:
        return (-1) * proportion_of_n * math.log2(proportion_of_n)

    if negative_examples == 0:
        return (-1) * proportion_of_p * math.log2(proportion_of_p)

    calculated_entropy = (-1) * proportion_of_p * math.log2(proportion_of_p) - proportion_of_n * math.log2(
        proportion_of_n)

    return calculated_entropy


# Given a set and a particular attribute, returns a set of all values the attribute
#   can take.
def all_known_attribute_values(data_set, attribute):

    known_values = set()

    for i in range(len(data_set['examples'])):
        if attribute in data_set['examples'][i].keys():
            known_values.add(data_set['examples'][i][attribute])

    if attribute is not 'label' and (len(known_values) == 1 and 1 in known_values):
        known_values.add(0)

    return known_values


# Given a data set, attribute, and value, produces a subset of examples with that attribute having
#   that particular value. Allows for sparse vectors and value = 0.
def subset(data_set, attribute, value):

    sub_set = {'examples': [], 'attributes': set()}

    for example in data_set['examples']:
        if attribute in example.keys():
            if example[attribute] == value:
                sub_set['examples'].append(example)
        else:
            if value == 0:
                sub_set['examples'].append(example)

    sub_set['attributes'] = data_set['attributes'].copy()
    sub_set['attributes'].remove(attribute)

    return sub_set


def common_label(data_set):

    labels_counts = {}
    max_label_count = 0
    max_label = None

    for example in data_set['examples']:
        label = example['label']
        if label in labels_counts.keys():
            labels_counts[label] += 1
        else:
            labels_counts[label] = 1
        if labels_counts[label] > max_label_count:
            max_label_count = labels_counts[label]
            max_label = label

    return max_label


if __name__ == '__main__':
    main()

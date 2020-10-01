from math import copysign
import random
import statistics


def main():

    training_set = read_file('data/csv-format/train.csv')
    test_set = read_file('data/csv-format/test.csv')
    learning_rates = [1, .1, .01]
    bias = random.uniform(-.01, .01)
    cv_epochs = 10
    training_epochs = 20

    weight_vector, new_bias, predictions, updates = perceptron(training_set['examples'], training_epochs,
                                                               len(training_set['features']), bias, learning_rates[2])
    predictions_on_test = predict_all(weight_vector, new_bias, test_set['examples'])
    training_acc = accuracy(predictions, training_set['examples'])
    test_acc = accuracy(predictions_on_test, test_set['examples'])
    #average_train = statistics.mean(training_acc)
    #average_test = statistics.mean(test_results[0.1])
    print(training_acc)
    print(test_acc)
    #print(updates)

    # folds = read_CV_folds_csv('data/csv-format/CVfolds/fold', 5)
    # training_results, test_results = CV_learning_rates(folds, cv_epochs, learning_rates, bias)
    # average_train = statistics.mean(training_results[0.1])
    # average_test = statistics.mean(test_results[0.1])
    # print(training_results)
    # print(average_train)

    return


def read_file(file_name):

    # Build dictionary data set from input file.
    data_set = {'examples': [], 'features': set()}

    # Read in CSV data formatted in sparse vector as: <label> <index1>:value1> <index2>:value2> ...
    with open(file_name, 'r') as f:
        for line in f.readlines():
            index = 0
            line = line.strip('\n')
            new_line = line.split(',')
            new_example = {'label': int(new_line[0]), 'features': []}
            for data in new_line[1:]:
                index += 1
                new_example['features'].append(float(data))
                data_set['features'].add(index)

            data_set['examples'].append(new_example)

    return data_set


def read_CV_folds_csv(file_name, number_of_folds):

    folds = []

    for i in range(1, number_of_folds+1):
        fold_name = file_name + str(i) + '.csv'
        fold = read_file(fold_name)
        folds.append(fold)

    return folds


def perceptron(data, epochs, features, bias=0, learning_rate=.1):

    weight_vector = weight_init(-.01, .01, features)
    updates = 0

    for epoch in range(epochs):
        random.shuffle(data)
        predictions = []
        for example in data:
            # Predict y' = sgn(w_t^T x_i)
            prediction = predict(weight_vector, example, bias)
            predictions.append(prediction)
            # If y' != y_i, update w_{t+1} = w_t + r(y_i x_i)
            if prediction != example['label']:
                updates += 1
                weight_vector, bias = update(weight_vector, bias, example, learning_rate, example['label'])

        acc = accuracy(predictions, data)
        # print(acc)

    return weight_vector, bias, predictions, updates


def CV_learning_rates(folds, epochs, learning_rates, bias):

    training_results = {}
    test_results = {}

    for rate in learning_rates:

        training_results[rate] = []
        test_results[rate] = []

        # For each heldout fold:
        for test_fold in range(len(folds)):
            all_but_one = {'examples': [], 'features': set()}
            for train_fold in range(len(folds)):
                if train_fold != test_fold:
                    all_but_one['examples'] += folds[train_fold]['examples']
                    all_but_one['features'] = all_but_one['features'].union(folds[train_fold]['features'])

            weights, bias, predictions, updates = perceptron(all_but_one['examples'], epochs, len(all_but_one['features']), bias=bias, learning_rate=rate)
            training_accuracy = accuracy(predictions, all_but_one['examples'])
            test_predictions = predict_all(weights, bias, folds[test_fold]['examples'])
            test_accuracy = accuracy(test_predictions, folds[test_fold]['examples'])

            training_results[rate].append(training_accuracy)
            test_results[rate].append(test_accuracy)

    return training_results, test_results


def weight_init(bottom, top, length):

    weights = []

    for i in range(length):
        rand_number = random.uniform(bottom, top)
        weights.append(rand_number)

    return weights


def predict(weights, example, bias=0):

    prediction = 0

    # Evaluate each feature to find the cross product of weights^T and example.
    for i in range(len(weights)):
        prediction += weights[i] * example['features'][i]

    return copysign(1, prediction + bias)


def predict_all(weights, bias, examples):

    predictions = []

    for example in examples:
        # Predict y' = sgn(w_t^T x_i)
        prediction = predict(weights, example, bias)
        predictions.append(prediction)

    return predictions


def update(weights, bias, x, rate, sign):

    new_weights = [0] * len(weights)
    new_bias = bias + rate * sign

    for index in range(len(weights)):
        new_weights[index] = weights[index] + (sign * rate * x['features'][index])

    return new_weights, new_bias


def accuracy(predictions, examples):

    summation = 0.0

    for index in range(len(predictions)):
        if predictions[index] == examples[index]['label']:
            summation += 1

    return summation / len(predictions)


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

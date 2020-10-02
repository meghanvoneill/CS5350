from math import copysign
import random
import matplotlib.pyplot as plt
import statistics


# Author: Meghan V. O'Neill
# For HW2: Perceptron & Variants
# Machine Learning, Fall 2020, Professor Srikumar, University of Utah

def main():

    random.seed(23)
    training_set = read_file('data/csv-format/train.csv')
    test_set = read_file('data/csv-format/test.csv')
    learning_rates = [1, .1, .01]
    bias = random.uniform(-.01, .01)
    cv_epochs = 10
    training_epochs = 20

    # Simple Perceptron
    weight_vector, new_bias, predictions, updates, accuracies = perceptron(training_set['examples'], training_epochs,
                                                               len(training_set['features']), bias, learning_rates[2])
    predictions_on_test = predict_all(weight_vector, new_bias, test_set['examples'])
    training_acc = accuracy(predictions, training_set['examples'])
    test_acc = accuracy(predictions_on_test, test_set['examples'])

    folds = read_CV_folds_csv('data/csv-format/CVfolds/fold', 5)
    training_results, test_results, cv_acc = CV_learning_rates(folds, cv_epochs, learning_rates, bias)
    average_train = statistics.mean(training_results[0.1])
    average_test = statistics.mean(test_results[0.1])
    print('Simple Perceptron\n')
    print('CV Epoch Accuracy by Learning Rate: ' + str(cv_acc))
    print('CV Test Accuracy by Learning Rate' + str(test_results))
    print('CV Average Test Accuracy: ' + str(average_test))
    print('Training Accuracy: ' + str(training_acc))
    print('Test Accuracy: ' + str(test_acc))
    print('Updates: ' + str(updates))
    print('\n')

    # Build x and y lists from epoch & accuracy data.
    x = []
    y = []

    for epoch in accuracies.keys():
        x.append(epoch)
        y.append(accuracies[epoch])

    # Plot data without alteration.
    plt.xlim(0, training_epochs + 1)
    plt.ylim(0, 1)
    plt.scatter(x, y, marker='o')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Simple Perceptron')
    plt.savefig('Simple_Perceptron.png')
    plt.show()



    # Decaying Learning Rate Perceptron
    weight_vector, new_bias, predictions, updates, accuracies = perceptron_decaying_learning_rate(training_set['examples'], training_epochs,
                                                               len(training_set['features']), bias, learning_rates[2])
    predictions_on_test = predict_all(weight_vector, new_bias, test_set['examples'])
    training_acc = accuracy(predictions, training_set['examples'])
    test_acc = accuracy(predictions_on_test, test_set['examples'])

    folds = read_CV_folds_csv('data/csv-format/CVfolds/fold', 5)
    training_results, test_results, cv_acc = CV_learning_rates(folds, cv_epochs, learning_rates, bias)
    average_train = statistics.mean(training_results[0.1])
    average_test = statistics.mean(test_results[0.1])
    print('Decaying Learning Rate Perceptron\n')
    print('CV Epoch Accuracy by Learning Rate: ' + str(cv_acc))
    print('CV Test Accuracy by Learning Rate' + str(test_results))
    print('CV Average Test Accuracy: ' + str(average_test))
    print('Training Accuracy: ' + str(training_acc))
    print('Test Accuracy: ' + str(test_acc))
    print('Updates: ' + str(updates))
    print('\n')

    # Build x and y lists from epoch & accuracy data.
    x = []
    y = []

    for epoch in accuracies.keys():
        x.append(epoch)
        y.append(accuracies[epoch])

    # Plot data without alteration.
    plt.xlim(0, training_epochs + 1)
    plt.ylim(0, 1)
    plt.scatter(x, y, marker='o')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Decaying Learning Rate Perceptron')
    plt.savefig('Decaying_Learning_Rate_Perceptron.png')
    plt.show()

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
    accuracies = {}

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
        accuracies[epoch + 1] = acc

    return weight_vector, bias, predictions, updates, accuracies


def perceptron_decaying_learning_rate(data, epochs, features, bias=0, learning_rate=.1):

    weight_vector = weight_init(-.01, .01, features)
    updates = 0
    accuracies = {}
    time_step = 0

    for epoch in range(epochs):
        random.shuffle(data)
        predictions = []
        lr = learning_rate / (1 + time_step)
        for example in data:
            # Predict y' = sgn(w_t^T x_i)
            prediction = predict(weight_vector, example, bias)
            predictions.append(prediction)
            # If y' != y_i, update w_{t+1} = w_t + r(y_i x_i)
            if prediction != example['label']:
                updates += 1
                weight_vector, bias = update(weight_vector, bias, example, lr, example['label'])

        acc = accuracy(predictions, data)
        accuracies[epoch + 1] = acc
        time_step += 1

    return weight_vector, bias, predictions, updates, accuracies


def perceptron_averaged(data, epochs, features, bias=0, learning_rate=.1):

    weight_vector = weight_init(-.01, .01, features)
    updates = 0
    accuracies = {}

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
        accuracies[epoch + 1] = acc

    return weight_vector, bias, predictions, updates, accuracies


def CV_learning_rates(folds, epochs, learning_rates, bias):

    training_results = {}
    test_results = {}
    accuracies = {}

    for rate in learning_rates:

        training_results[rate] = []
        test_results[rate] = []
        accuracies[rate] = []

        # For each heldout fold:
        for test_fold in range(len(folds)):
            all_but_one = {'examples': [], 'features': set()}
            for train_fold in range(len(folds)):
                if train_fold != test_fold:
                    all_but_one['examples'] += folds[train_fold]['examples']
                    all_but_one['features'] = all_but_one['features'].union(folds[train_fold]['features'])

            weights, bias, predictions, updates, acc = perceptron(all_but_one['examples'], epochs, len(all_but_one['features']), bias=bias, learning_rate=rate)
            training_accuracy = accuracy(predictions, all_but_one['examples'])
            test_predictions = predict_all(weights, bias, folds[test_fold]['examples'])
            test_accuracy = accuracy(test_predictions, folds[test_fold]['examples'])

            training_results[rate].append(training_accuracy)
            test_results[rate].append(test_accuracy)
            accuracies[rate].append(acc)

    return training_results, test_results, accuracies


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

import math


def main():

    match_lecture_notes()
    #match_quiz_data()

    # Question 18: Testing
    # tests = [([-1, -1, -1], -1), ([-1, 1, -1], -1), ([1, -1, -1], -1), ([1, 1, -1], 1)]
    # predictions = []
    #
    # for test in tests:
    #     hypothesis = 0
    #     classifiers_results = test[0]
    #     label = test[1]
    #     for index, result in enumerate(classifiers_results):
    #         hypothesis += result * hypothesis_weights[index]
    #     prediction = math.copysign(1, hypothesis)
    #     predictions.append(prediction)
    #
    # print('18. Predictions: ' + str(predictions))

    return


def match_quiz_data():

    # hypotheses for examples = [(hypothesis, label)_1, (hypothesis, label)_2, ... (hypothesis, label)_m]
    #   m = 4
    classifier_1_hypotheses = [(-1, -1), (-1, -1), (1, -1), (1, 1)]
    classifier_2_hypotheses = [(-1, -1), (1, -1), (-1, -1), (1, 1)]
    classifier_3_hypotheses = [(-1, -1), (-1, -1), (-1, -1), (-1, 1)]
    all_classifier_results = [classifier_1_hypotheses, classifier_2_hypotheses, classifier_3_hypotheses]

    # Run AdaBoost to get the alpha weights for each classifier: T = 3.
    T = 3
    hypothesis_weights = ada_boost(all_classifier_results, T)


def match_lecture_notes():

    # hypotheses for examples = [(hypothesis, label)_1, (hypothesis, label)_2, ... (hypothesis, label)_m]
    #   m = 4
    classifier_1_hypotheses = [(1, 1), (1, 1), (-1, -1), (-1, -1), (-1, -1), (-1, 1), (-1, 1), (-1, 1), (-1, -1), (-1, -1)]
    classifier_2_hypotheses = [(1, 1), (1, 1), (1, -1), (1, -1), (1, -1), (1, 1), (1, 1), (1, 1), (-1, -1), (-1, -1)]
    classifier_3_hypotheses = [(-1, 1), (-1, 1), (-1, -1), (-1, -1), (-1, -1), (1, 1), (1, 1), (1, 1), (1, -1), (-1, -1)]
    all_classifier_results = [classifier_1_hypotheses, classifier_2_hypotheses, classifier_3_hypotheses]

    # Run AdaBoost to get the alpha weights for each classifier: T = 3.
    T = 3
    hypothesis_weights = ada_boost(all_classifier_results, T)


def ada_boost(training_set, T_rounds):

    m_size = len(training_set[0])
    init_weight = 1 / float(m_size)
    D_weights = [init_weight] * m_size

    classifiers_available = set()
    for i in range(len(training_set)):
        classifiers_available.add(i)

    for round_t in range(T_rounds):
        # Find the classifier whose weighted classification error is better than chance.
        classifier_index, epsilon_t = choose_best_classifier_or_numerically(training_set, classifiers_available, D_weights)
        classifiers_available.remove(classifier_index)
        print('classifier index: ' + str(classifier_index))
        print('epsilon_t: ' + str(epsilon_t))

        # Compute its vote:
        #   alpha_t = 1/2 * ln[(1 - epsilon_t) / epsilon_t]
        alpha_t = calculate_alpha_value(epsilon_t)
        print('alpha_t: ' + str(alpha_t))

        # Update the values of the weights for the training examples:
        #   D_{t + 1}(i) = (D_t(i) / Z_t) * e ^ (-alpha_t * y_i * h_t(x_i))
        D_weights = update_weights(D_weights, alpha_t, training_set[classifier_index])

    return D_weights


#   epsilon_t = 1/2 - 1/2 * summation [D_t(i) * y_i * h(x_i)] from i = 0 to m
def calculate_weighted_error(D_weights, hypotheses):

    summation = 0

    # For each hypothesis made by this classifier, sum each hypothesis's sign with the corresponding
    # weight from D_weights.
    for i in range(len(hypotheses)):
        hypothesis_results = hypotheses[i]
        hypothesis_sign = int(hypothesis_results[0]) * int(hypothesis_results[1])
        weighted_error = D_weights[i] * hypothesis_sign
        summation += weighted_error

    epsilon_t = (1/2.) - ((1/2.) * summation)
    return epsilon_t


#   alpha_t = 1/2 * ln[(1 - epsilon_t) / epsilon_t]
def calculate_alpha_value(epsilon_t):

    weighting_of_error = (1 - epsilon_t) / epsilon_t
    alpha_t = (1/2) * math.log(weighting_of_error)

    return alpha_t


#   D_{t + 1}(i) = (D_t(i) / Z_t) * e ^ (-alpha_t * y_i * h_t(x_i))
def update_weights(D_weights, alpha, hypotheses):

    new_weights = []
    running_weights_total = 0

    print('weights before: ' + str(D_weights))

    # For each hypothesis and weight, update the weight.
    for index in range(len(hypotheses)):
        # Calculate new weight based on previous weight.
        hypothesis_results = hypotheses[index]
        hypothesis_sign = int(hypothesis_results[0]) * int(hypothesis_results[1])
        exponent = - alpha * hypothesis_sign
        weight = D_weights[index] * math.pow(math.e, exponent)
        new_weights.append(weight)

        running_weights_total += weight

    print('updated weight total: ' + str(running_weights_total))
    print(new_weights)

    # Normalize the weights to total to 1.
    for index in range(len(new_weights)):
        w = new_weights[index] / running_weights_total
        new_weights[index] = w

    print('weights after: ' + str(new_weights))

    return new_weights


def choose_best_classifier_or_numerically(classifiers_data, classifiers_available, D_weights):

    index_of_best = -1
    lowest_epsilon = 1
    # accuracy_of_best = -1

    # For each classifier:
    for index, classifier_results in enumerate(classifiers_data):
        # If this classifier is not available, continue to the next.
        if index not in classifiers_available:
            continue
        # Calculate the weighted error for this classifier.
        epsilon = calculate_weighted_error(D_weights, classifier_results)

        if epsilon < lowest_epsilon:
            lowest_epsilon = epsilon
            index_of_best = index

        # # Look at this classifier's prediction for each example
        # correct = 0
        # count = 0
        # for result in classifier_results:
        #     guess = result[0]
        #     label = result[1]
        #     if guess == label:
        #         correct += 1
        #     count += 1
        # accuracy = correct / count
        # if accuracy > accuracy_of_best:
        #     accuracy_of_best = accuracy
        #     index_of_best = index
    # print('acc: ' + str(1 - accuracy_of_best))

    return index_of_best, epsilon


if __name__ == '__main__':
    main()

from math import copysign


def main():

    training_set = read_file('data/csv-format/train.csv')

    weight_vector, predictions, updates = perceptron(training_set)

    print(weight_vector)
    print(predictions)
    print(updates)

    acc = accuracy(predictions, training_set['examples'])

    print(acc)

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


def perceptron(data, bias=0, learning_rate=.1):

    weight_vector = [0] * len(data['features'])
    predictions = []
    updates = 0

    for example in data['examples']:
        # Predict y' = sgn(w_t^T x_i)
        prediction = predict(weight_vector, example, bias)
        predictions.append(prediction)
        # If y' != y_i, update w_{t+1} = w_t + r(y_i x_i)
        if prediction != example['label']:
            updates += 1
            weight_vector = update(weight_vector, example, learning_rate, example['label'])

    return weight_vector, predictions, updates


def predict(weights, example, bias=0):

    prediction = 0

    # Evaluate each feature to find the cross product of weights^T and example.
    for i in range(len(weights)):
        prediction += weights[i] * example['features'][i]

    return copysign(1, prediction + bias)


def update(weights, x, rate, sign):

    new_weights = [0] * len(weights)

    for index in range(len(weights)):
        new_weights[index] = weights[index] + (sign * rate * x['features'][index])

    return new_weights


def accuracy(predictions, examples):

    summation = 0.0

    for index in range(len(predictions)):
        if predictions[index] == examples[index]['label']:
            summation += 1

    return summation / len(predictions)


if __name__ == '__main__':
    main()

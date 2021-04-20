from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    feature_vector = np.array(feature_vector)
    theta = np.array(theta)
    z = label * (np.dot(theta, feature_vector) + theta_0)
    if z >= 1:
        return 0
    else:
        return 1 - z


def hinge_loss_full(feature_matrix, labels, theta, theta_0):

    h = np.dot(feature_matrix, np.transpose(theta)) + theta_0
    length = np.size(h)
    sum_loss = 0
    count = 0
    for i in range(length):
        z = labels[i] * h[i]
        if z < 1:
            sum_loss = sum_loss + (1 - z)
        count += 1
    loss = sum_loss / length
    return loss


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 0:
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label
    res = (current_theta, current_theta_0)
    return res


def perceptron(feature_matrix, labels, T):

    (n_samples, n_features) = feature_matrix.shape
    theta = np.zeros((n_features,), dtype=int)
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(np.transpose(feature_matrix[i]), labels[i], theta, theta_0)
    res = (theta, theta_0)
    return res


def average_perceptron(feature_matrix, labels, T):

    (n_samples, n_features) = feature_matrix.shape
    theta = np.zeros((n_features,), dtype=int)
    theta_0 = 0
    sum_theta = 0
    sum_theta_0 = 0
    n = feature_matrix.shape[0]
    res = ()
    for t in range(T):
        for i in get_order(n):
            theta, theta_0 = perceptron_single_step_update(np.transpose(feature_matrix[i]), labels[i], theta, theta_0)
            sum_theta += theta
            sum_theta_0 += theta_0
    res = (sum_theta / (n*T), sum_theta_0 / (n*T))
    return res


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):

    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1:
        current_theta = current_theta * (1 - eta * L) + feature_vector * eta * label
        current_theta_0 += eta * label
    else:
        current_theta = (1 - eta * L) * current_theta
    res = (current_theta, current_theta_0)
    return res


def pegasos(feature_matrix, labels, T, L):

    n_samples, n_features = feature_matrix.shape
    current_theta = np.zeros((n_features, ), dtype=int)
    current_theta_0 = 0
    counter = 0
    for t in range(1, T + 1):
        for i in get_order(n_samples):
            counter += 1
            eta = 1 / (np.sqrt(counter))
            current_theta, current_theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta,
                                                                        current_theta, current_theta_0)
    return current_theta, current_theta_0


# Part II


def classify(feature_matrix, theta, theta_0):

    n = feature_matrix.shape[0]
    labels = np.zeros((n, ), dtype=int)
    for i in range(n):
        if (np.dot(theta, feature_matrix[i]) + theta_0) > 0:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_classifier_labels = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_classifier_labels, train_labels)

    val_classifier_labels = classify(val_feature_matrix, theta, theta_0)
    val_accuracy = accuracy(val_classifier_labels, val_labels)
    return train_accuracy, val_accuracy


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # my code here
    dictionary = {}     # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):
    return (preds == targets).mean()

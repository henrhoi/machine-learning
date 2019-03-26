from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import math

"""
Decision Tree Learning Algorithm
@author: henrhoi
"""


class Node:
    """
    Class for representing a node in the Decision Tree
    """

    def __init__(self, attribute, branches=None):
        if branches is None:
            branches = {}
        self.attribute = attribute
        self.branches = branches

    def add_branch(self, label, subtree):
        self.branches[label] = subtree

    def match(self, value):
        try:
            return self.branches[value]

        except KeyError:
            raise ValueError("Not valid value for decision")


def decision_tree_learning(examples, attributes, parent_examples):
    """
    Decision Tree Learning
    :param examples: Dataframe with training data on current level
    :param attributes: List of possible attributes in examples
    :param parent_examples: Dataframe with training data on previous level
    :return: Root to current tree
    """

    if not examples.count:
        return plurality_value(parent_examples)
    elif len(examples['Class'].value_counts()) == 1 or not len(attributes):
        return plurality_value(examples)

    else:
        importances = importance(examples, attributes, "info_gain")
        max_index = importances.index(max(importances))  # Choosing first since it is arbitrary
        attribute = attributes[max_index]
        tree = Node(attribute)

        for value in examples[attribute].value_counts().keys().tolist():
            next_examples = examples.loc[examples[attribute] == value]
            next_attributes = [attr for attr in attributes if attr != attribute]
            subtree = decision_tree_learning(next_examples, next_attributes, examples)
            tree.add_branch(value, subtree)
        return tree


def plurality_value(examples):
    """
    Plurality value - most common class in examples
    """

    counts = examples['Class'].value_counts()
    if len(counts) == 2 and counts.index[0] == counts.index[1]:
        return counts.index[random.choice([0, 1])]

    return counts.index[0]


def importance(examples, attributes, importance_type):
    """
    Calculating importance for all attributes based on examples
    """

    if importance_type == "random":
        return random_importance(attributes)

    elif importance_type == "info_gain":
        return information_gain(examples, attributes)

    else:
        raise ValueError("Importance type must be 'info_gain' or 'random'")


def random_importance(attributes):
    """
    Generating random importances
    """
    return [random.uniform(0, 1) for _ in range(len(attributes))]


def information_gain(examples, attributes):
    """
    Calculating information gain based on improvement of entropy
    """

    def entropy(q):
        if q in [0.0, 1.0]:
            return 0

        return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))

    def remainder():
        result = 0
        subsets = chunk_dataframe(examples, attribute)
        for subset in subsets:
            subset_count_dict = get_count_dict(subset)
            p_k, n_k = subset_count_dict[2], subset_count_dict[1]
            result += ((p_k + n_k) / (sum(count_dict.values()))) * entropy(p_k / (p_k + n_k))

        return result

    importances = []

    # Calculating importance for each attribute
    for attribute in attributes:
        count_dict = get_count_dict(examples)
        attribute_entropy = entropy(count_dict[2] / (count_dict[1] + count_dict[2]))
        attribute_remainder = remainder()
        importances.append(attribute_entropy - attribute_remainder)

    print(importances)
    return importances


def predict(decision_tree, test):
    """
    Predicting data based on already built decision tree
    :param decision_tree: Decision tree built with {Node} objects
    :param test: Dataframe with data
    :return: List of predicitions
    """
    predictions = []
    for index, row in test.iterrows():
        predictions.append(predict_row(decision_tree, row))

    return predictions


def predict_row(decision_tree, row):
    """
    Predicts one row from dataframe
    """
    attribute = decision_tree.attribute
    attribute_value = row[attribute]
    subtree = decision_tree.match(attribute_value)
    return predict_row(subtree, row) if isinstance(subtree, Node) else subtree


"""
Utils - Methods used for minor tasks
"""


def chunk_dataframe(df, attribute):
    """
    Chunks dataframe into n parts based on attribute-values
    :param df: Dataframe for separation
    :param attribute : Attribute to split on
    """
    return [df_attribute for _, df_attribute in df.groupby(attribute)]


def get_count_dict(dataframe):
    """
    Returnes series with counts for values, if no occurrence count will be one
    :param dataframe: Dataframe with data
    :param attribute: Attribute
    :return: Dictionary with values pointing to occurrences
    """
    default_values = [1, 2]
    count_df = dataframe['Class'].value_counts()
    count_dict = dict(zip(count_df.keys().tolist(), count_df.tolist()))
    for value in default_values:
        if value not in count_dict:
            count_dict[value] = 0
    return count_dict


def get_datasets():
    """
    Reading datasets with Pandas
    """
    features = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Class"]
    training = pd.read_csv('data/training.txt', sep="	", header=None)
    training.columns = features

    test = pd.read_csv('data/test.txt', sep="	", header=None)
    test.columns = features

    return training, test


def print_tree(tree, value="", level=0):
    """
    Prints tree structure
    """
    print("\t" * level + str(value), end=" -> ")
    attribute = tree.attribute if isinstance(tree, Node) else str(tree)
    print(attribute, end="\n")

    if isinstance(tree, Node):
        for label, subtree in tree.branches.items():
            print_tree(subtree, label, level + 1)


def draw_correlation_map(data, save=False):
    """
    Draws correlation map of features in dataframe
    """

    colormap = sns.diverging_palette(100, 250, as_cmap=True)
    plt.figure(figsize=(10, 10))
    plt.title('Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.select_dtypes([np.number]).astype(float).corr(), linewidths=0.1, vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True)

    plt.savefig('correlation_map.png') if save else None
    plt.show()


def main():
    """
    Reading data into dataframes
    Assumptions:
        - Assuming value 2 is positive and 1 is negative
        - Labels have been written in text, e.g. 1 --> 'One'
    """

    training, test = get_datasets()
    draw_correlation_map(training, save=False)

    attributes = list(training.columns)
    attributes.remove("Class")

    # Building decision tree
    decision_tree = decision_tree_learning(training, attributes, training)

    # Printing the Decision Tree
    print("Decision tree:")
    print_tree(decision_tree)

    # Splitting x and y from test data
    test_y = test["Class"]
    test_x = test.drop("Class", axis=1)

    # Predicting test data
    predictions = predict(decision_tree, test_x)
    accuracy = accuracy_score(test_y, predictions)
    print("Accuracy:", accuracy)

    # Saving predictions to file
    pred_df = pd.DataFrame(predictions, columns=['Predictions'])
    pred_df.to_csv("data/predictions.csv", index=False)


main()

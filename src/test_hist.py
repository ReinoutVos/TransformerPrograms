import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import hist

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"


def make_hist(vocab_size, dataset_size, min_length=2, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 2)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        counts = Counter(sent)
        sents.append([BOS] + sent)
        tags.append([PAD] + [str(counts[c]) for c in sent])
    return pd.DataFrame({"sent": sents, "tags": tags})



def create_dataset(upTo=32, vocab_size=8, dataset_size=5):
    """
    Create a dataset for the sort task.

    Returns:
        tuple: A tuple containing two lists: x (input sequences) and y (expected output sequences).
    """
    x = []
    y = []
    for i in range(1, upTo):
        dataset = make_hist(vocab_size, dataset_size, i, i+1)
        ip = dataset['sent']
        op = dataset['tags']
        
        x.append(ip)
        y.append(op)

    return np.array(x), np.array(y)

def calculate_accuracy(y_pred, y_true):
    """
    Calculate the accuracy of predicted sequences.

    Args:
        y_pred (list): The predicted sequence.
        y_true (list): The true sequence.

    Returns:
        float: The accuracy of the predicted sequence.
    """
    if len(y_pred) != len(y_true):
        raise ValueError("The lengths of y_pred and y_true lists are not the same.")

    # Remove first element from each list
    y_pred = [pred[1:] for pred in y_pred]
    y_true = [true[1:] for true in y_true]

    # print("preds", y_pred)
    # print("tags", y_true)  

    correct_y_predictions = 0

    for pred, truth in zip(y_pred, y_true):
        if pred == truth:
            correct_y_predictions += 1

    accuracy = correct_y_predictions / len(y_pred)
    return accuracy


def length_generalization():
    """
    Perform length generalization on the sort task.

    Returns:
        dict: A dictionary containing the accuracies for different sequence lengths.
    """
    x, y = create_dataset(upTo=3, vocab_size=8)
    accuracies = {}

    for i, (sentences, tags) in enumerate(zip(x, y), start=1):
        preds = [hist.run(sent) for sent in sentences]
        accuracies[i] = calculate_accuracy(preds, tags)

    return accuracies

def plot_acc(data):
    """
    Plot the accuracy of the length generalization on the sort task.

    Args:
        data (dict): A dictionary containing the accuracies for different sequence lengths.
    """
    x = list(data.keys())
    y = list(data.values())
    plt.plot(x, y)
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Length Generalization on the sort task')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    acc = length_generalization()
    plot_acc(acc)
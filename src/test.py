import numpy as np
import pandas as pd
import sort
import matplotlib.pyplot as plt

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"

def make_sort(vocab_size, dataset_size, min_length=4, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 3)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length - 1)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        sents.append([BOS] + sent + [EOS])
        tags.append([PAD] + sorted(sent) + [PAD])
    return pd.DataFrame({"sent": sents, "tags": tags})


def create_dataset():
    x = []
    y = []
    for i in range(1, 31):
        dataset = make_sort(32, 5, i, i+2)
        ip = dataset['sent'].tolist()
        op = dataset['tags'].tolist()
        x.append(ip)
        y.append(op)
    return x, y

def calculate_accuracy(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("The lengths of y_pred and y_true lists are not the same.")

    y_pred = y_pred[1:-1]
    y_true = y_true[1:-1]
    correct_y_predictions = 0
    total_positions = len(y_pred)

    for i in range(total_positions):
        if y_pred[i] == y_true[i]:
            correct_y_predictions += 1

    accuracy = (correct_y_predictions) / (total_positions)
    return accuracy

def length_generalization():
    x, y = create_dataset()
    accuracies = {}
    for i in range(len(x)):
        # Accuracy sum for sequences with a length i+4
        acc_sum = 0
        for (x_i, y_i) in list(zip(x[i], y[i])):
            y_pred = sort.run(x_i)
            acc_sum += calculate_accuracy(y_pred, y_i)
            # if (i == 4):
            #     print(f"Input : {x_i}\nExpected : {y_i}\nPredicted : {y_pred}")
            #     print(acc_sum)
            #     raise ValueError
        accuracies[f"{i+3}"] = (acc_sum / len(x[i]))
    return accuracies


def plot_acc(data):
    x = list(map(int, data.keys()))
    y = list(data.values())
    plt.plot(x, y)
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Length Generalization on the sort task')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    acc = length_generalization()
    # print(acc)
    plot_acc(acc)
import numpy as np
import pandas as pd
import sort
import matplotlib.pyplot as plt

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"

# TO TEST YOUR MODEL, COPY THE SORT.PY FILE FROM THE OUTPUT FOLDER AND PASTE THE CODE INTO THE SORT.PY FILE IN THE SRC FOLDER

# Function directly copied from src/utils/data_utils.py
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


def create_dataset(min_seq_len=1, max_seq_len=31):
    x = []
    y = []
    for i in range(min_seq_len, max_seq_len):
        dataset = make_sort(32, 100, i, i+2)
        ip = dataset['sent'].tolist()
        op = dataset['tags'].tolist()
        x.append(ip)
        y.append(op)
    return x, y

# Calculates accuracy as the correct places / total places (expected = 1,2,3,4 | output = 1, 2, 4, 3 | acc = 0.5)
def calculate_accuracy(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("The lengths of y_pred and y_true lists are not the same.")

    # Ignores the starting and ending characters (<s> and </s>)
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
        accuracies[f"{i+3}"] = (acc_sum / len(x[i]))
    return accuracies


def plot_acc(data, filepath):
    x = list(map(int, data.keys()))
    y = list(data.values())
    plt.plot(x, y)
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Length Generalization on the sort task')
    plt.grid(True)
    # plt.savefig('length_gen/sort_1_8_cat.png')
    plt.savefig(filepath)
    plt.show()


if __name__ == '__main__':
    acc = length_generalization()

    '''
    Saves the dictionary as a text file
    Give the same name as the saved accuracy plot from plot_acc
    Add the following data to the txt file:
    1. Layers 
    2. Attention Heads (If both categorical and numerical used, breakdown of that)
    3. MLPs (If both categorical and numerical used, breakdown of that)
    4. On what sequence length it was trained on

    (Instead of 1,2,3 ; you can also write the table from which you have chosen the parameters)
    '''
    with open("length_gen/sort_cat_8_16.txt", 'w') as f:  
        f.write("Length Generalization for model trained on sequence lengths of 8-16 using only categorical attention and mlps. (Table 3 hyperparams used)\n")
        for key, value in acc.items():  
            f.write('%s : %s\n' % (key, value))
    # print(acc)
    plot_acc(acc, 'length_gen/sort_cat_8_16.png')
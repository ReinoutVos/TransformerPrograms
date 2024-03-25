import argparse

from utils import data_utils, metric_utils
from sort import run
import random

import sys


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=8)
    
    args = parser.parse_args()

    return args

def create_dataset(maxlen, samples):
    x = []
    y = []

    MAX_LENGTH = maxlen - 2

    for i in range(samples):


        list1 = [(random.randint(0, MAX_LENGTH-2)) for _ in range(MAX_LENGTH)]

        list2 = list1.copy()
        list2.sort()

        list1 = ['<s>'] + list(map(str, list1)) + ['</s>']
        list2 = ['<s>'] + list(map(str, list2)) + ['</s>']
        
        x.append(list1)
        y.append(list2)

    return x, y

if __name__ == "__main__":
    args = parse_args()
    x, y = create_dataset(args.max_length, args.dataset_size)

    for (i, j) in list(zip(x ,y)):
        print(f"Input : {i}\nExpected : {j}\nPredicted : {run(i)}\n\n")

    print(run(["<s>", "9", "1", "6", "</s>"]))
    print(run(["<s>", "8", "4", "4", "0", "6", "9", "</s>"]))
    print(run(["<s>", "11", "4", "1", "0", "6", "6", "5", "1", "3", "</s>"]))

    
'''
python3 src/test.py --dataset_size 10 --max_length 8;
'''


'''
vocab size : decides what tokens and position the model can look at. for ex: if the vocab size is 16, positions: 0-15, tokens: <s>, </s>, <pad>, 0-(vocab_size-4) (0-12)
min_length : minimum length of the sequence that the dataset is trained on
max_length : maximum length of the sequence that the dataset is trained on

To show for length generalization:
train on say, min_length 8, max_length 16
test on 1-100, for example
Accuracy graph should be like a plateau, accuracy will be high only on sequences length that it was trained on

'''
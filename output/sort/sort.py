import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/sort/sort_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 2
        elif q_position in {8, 1, 13}:
            return k_position == 5
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7, 9, 10, 11, 12, 15}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 12

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3, 14}:
            return k_position == 4
        elif q_position in {4, 12}:
            return k_position == 5
        elif q_position in {5, 7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {11, 13}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 13

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 6}:
            return k_position == 3
        elif q_position in {3, 14}:
            return k_position == 4
        elif q_position in {9, 13, 4, 12}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 11}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 13

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 14}:
            return token == "7"
        elif position in {1, 15}:
            return token == "11"
        elif position in {2}:
            return token == "0"
        elif position in {3, 7}:
            return token == "1"
        elif position in {8, 10, 4, 5}:
            return token == "5"
        elif position in {13, 6}:
            return token == "9"
        elif position in {9}:
            return token == "6"
        elif position in {11, 12}:
            return token == "8"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 8, 5, 9}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 4, 12, 7}:
            return k_position == 2
        elif q_position in {10, 13, 6}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 11

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 2
        elif q_position in {1, 4, 7, 8, 11, 13}:
            return k_position == 6
        elif q_position in {2, 14}:
            return k_position == 4
        elif q_position in {9, 10, 3, 12}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 13

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 5, 6}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 14}:
            return k_position == 4
        elif q_position in {11, 3, 4, 13}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9, 10, 12}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 2, 3}:
            return token == "0"
        elif position in {4, 5}:
            return token == "7"
        elif position in {12, 6}:
            return token == "8"
        elif position in {15, 7}:
            return token == "11"
        elif position in {8}:
            return token == "4"
        elif position in {9, 10, 11, 13}:
            return token == "9"
        elif position in {14}:
            return token == "<pad>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {3, 9, 11, 12, 13}:
            return 9
        elif key in {2, 10}:
            return 8
        elif key in {0, 14}:
            return 12
        elif key in {1}:
            return 2
        return 4

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {1, 2, 8, 10}:
            return 10
        elif key in {6, 11, 12, 13}:
            return 12
        elif key in {9}:
            return 7
        elif key in {3}:
            return 11
        return 5

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_7_output, position):
        key = (attn_0_7_output, position)
        if key in {
            ("1", 0),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 7),
            ("10", 8),
            ("10", 9),
            ("10", 10),
            ("10", 11),
            ("10", 12),
            ("10", 13),
            ("10", 14),
            ("10", 15),
            ("11", 2),
            ("11", 4),
            ("11", 7),
            ("11", 8),
            ("11", 10),
            ("11", 12),
            ("11", 14),
            ("2", 2),
            ("2", 3),
            ("2", 7),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 15),
            ("3", 2),
            ("3", 7),
            ("3", 15),
            ("4", 2),
            ("4", 7),
            ("4", 9),
            ("4", 15),
            ("5", 2),
            ("5", 7),
            ("5", 9),
            ("5", 15),
            ("6", 2),
            ("6", 7),
            ("6", 8),
            ("6", 9),
            ("6", 10),
            ("6", 15),
            ("7", 2),
            ("7", 7),
            ("7", 9),
            ("7", 11),
            ("7", 15),
            ("8", 2),
            ("8", 3),
            ("8", 4),
            ("8", 7),
            ("8", 8),
            ("8", 10),
            ("8", 12),
            ("8", 14),
            ("9", 2),
            ("9", 7),
            ("9", 15),
            ("</s>", 2),
            ("</s>", 7),
            ("</s>", 15),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 14),
            ("<s>", 15),
        }:
            return 15
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("1", 1),
            ("10", 1),
            ("11", 1),
            ("11", 3),
            ("11", 9),
            ("11", 11),
            ("11", 15),
            ("12", 1),
            ("12", 2),
            ("12", 7),
            ("12", 9),
            ("12", 11),
            ("12", 15),
            ("2", 1),
            ("3", 1),
            ("3", 9),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("8", 1),
            ("8", 9),
            ("8", 11),
            ("8", 15),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 14
        elif key in {
            ("10", 6),
            ("11", 6),
            ("12", 6),
            ("2", 6),
            ("3", 6),
            ("4", 6),
            ("5", 6),
            ("6", 6),
            ("7", 6),
            ("8", 6),
            ("9", 6),
            ("</s>", 6),
            ("<s>", 6),
        }:
            return 0
        elif key in {("0", 6)}:
            return 8
        return 5

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_7_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position, attn_0_1_output):
        key = (position, attn_0_1_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "10"),
            (0, "11"),
            (0, "12"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "6"),
            (0, "7"),
            (0, "8"),
            (0, "9"),
            (0, "</s>"),
            (0, "<s>"),
            (2, "</s>"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "</s>"),
            (5, "0"),
            (5, "1"),
            (5, "10"),
            (5, "11"),
            (5, "12"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "10"),
            (6, "11"),
            (6, "12"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "6"),
            (6, "7"),
            (6, "8"),
            (6, "9"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "</s>"),
            (8, "</s>"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "</s>"),
            (12, "2"),
            (12, "</s>"),
            (12, "<s>"),
            (13, "2"),
            (13, "</s>"),
            (13, "<s>"),
            (14, "10"),
            (14, "11"),
            (14, "12"),
            (14, "2"),
            (14, "3"),
            (14, "5"),
            (14, "6"),
            (14, "7"),
            (14, "8"),
            (14, "9"),
            (14, "</s>"),
            (14, "<s>"),
            (15, "10"),
            (15, "2"),
            (15, "7"),
            (15, "8"),
            (15, "</s>"),
            (15, "<s>"),
        }:
            return 15
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "10"),
            (1, "11"),
            (1, "12"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "<s>"),
            (2, "0"),
            (2, "12"),
            (3, "0"),
            (4, "0"),
            (7, "0"),
            (7, "10"),
            (7, "11"),
            (7, "12"),
            (7, "2"),
            (8, "0"),
            (8, "10"),
            (9, "0"),
            (10, "0"),
            (11, "0"),
            (11, "12"),
            (12, "0"),
            (12, "12"),
            (15, "0"),
        }:
            return 12
        elif key in {
            (2, "2"),
            (3, "10"),
            (3, "11"),
            (3, "12"),
            (3, "2"),
            (8, "2"),
            (9, "12"),
            (9, "2"),
            (10, "12"),
            (11, "2"),
            (15, "12"),
        }:
            return 0
        elif key in {(3, "7"), (3, "8")}:
            return 10
        elif key in {(8, "12")}:
            return 4
        return 13

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(positions, attn_0_1_outputs)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 2
        elif q_position in {1, 10, 6}:
            return k_position == 3
        elif q_position in {2, 3}:
            return k_position == 14
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {8, 9, 12, 7}:
            return k_position == 5
        elif q_position in {11, 13, 14, 15}:
            return k_position == 4

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_6_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 6}:
            return token == "0"
        elif mlp_0_1_output in {1, 3, 12}:
            return token == "1"
        elif mlp_0_1_output in {4, 7, 8, 9, 10, 13}:
            return token == "</s>"
        elif mlp_0_1_output in {5}:
            return token == "5"
        elif mlp_0_1_output in {11, 14}:
            return token == "<s>"
        elif mlp_0_1_output in {15}:
            return token == "4"

    attn_1_1_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, mlp_0_3_output):
        if mlp_0_1_output in {0, 10}:
            return mlp_0_3_output == 12
        elif mlp_0_1_output in {1}:
            return mlp_0_3_output == 1
        elif mlp_0_1_output in {2}:
            return mlp_0_3_output == 3
        elif mlp_0_1_output in {3, 4}:
            return mlp_0_3_output == 2
        elif mlp_0_1_output in {5}:
            return mlp_0_3_output == 10
        elif mlp_0_1_output in {13, 6, 7}:
            return mlp_0_3_output == 5
        elif mlp_0_1_output in {8}:
            return mlp_0_3_output == 6
        elif mlp_0_1_output in {9, 11}:
            return mlp_0_3_output == 15
        elif mlp_0_1_output in {12}:
            return mlp_0_3_output == 14
        elif mlp_0_1_output in {14, 15}:
            return mlp_0_3_output == 0

    attn_1_2_pattern = select_closest(mlp_0_3_outputs, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_5_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, attn_0_0_output):
        if position in {0, 2, 10}:
            return attn_0_0_output == "10"
        elif position in {1, 7}:
            return attn_0_0_output == "0"
        elif position in {3}:
            return attn_0_0_output == "12"
        elif position in {4, 14}:
            return attn_0_0_output == "8"
        elif position in {5}:
            return attn_0_0_output == "7"
        elif position in {6, 15}:
            return attn_0_0_output == "9"
        elif position in {8, 11, 13}:
            return attn_0_0_output == "6"
        elif position in {9}:
            return attn_0_0_output == "2"
        elif position in {12}:
            return attn_0_0_output == "11"

    attn_1_3_pattern = select_closest(attn_0_0_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 1
        elif q_position in {1, 14}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {9, 3, 12, 13}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5, 6}:
            return k_position == 2
        elif q_position in {8, 15}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 13

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {8, 10, 11, 13, 15}:
            return k_position == 9
        elif q_position in {9, 12}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 13

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 3, 5}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 12}:
            return k_position == 4
        elif q_position in {10, 4}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {8, 7}:
            return k_position == 7
        elif q_position in {9, 13}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 15

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"0", "10"}:
            return attn_0_2_output == "1"
        elif attn_0_0_output in {"1"}:
            return attn_0_2_output == "0"
        elif attn_0_0_output in {"11"}:
            return attn_0_2_output == "10"
        elif attn_0_0_output in {"12"}:
            return attn_0_2_output == "7"
        elif attn_0_0_output in {"2"}:
            return attn_0_2_output == "5"
        elif attn_0_0_output in {"3"}:
            return attn_0_2_output == "2"
        elif attn_0_0_output in {"4", "6"}:
            return attn_0_2_output == "3"
        elif attn_0_0_output in {"5"}:
            return attn_0_2_output == "<s>"
        elif attn_0_0_output in {"7", "9"}:
            return attn_0_2_output == "</s>"
        elif attn_0_0_output in {"8"}:
            return attn_0_2_output == "11"
        elif attn_0_0_output in {"</s>"}:
            return attn_0_2_output == "4"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_2_output == "<pad>"

    attn_1_7_pattern = select_closest(attn_0_2_outputs, attn_0_0_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(position, attn_1_7_output):
        key = (position, attn_1_7_output)
        if key in {
            (6, "0"),
            (6, "1"),
            (6, "10"),
            (6, "11"),
            (6, "12"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "6"),
            (6, "7"),
            (6, "8"),
            (6, "9"),
            (6, "</s>"),
            (6, "<s>"),
            (14, "2"),
            (14, "3"),
            (14, "6"),
            (14, "9"),
            (15, "3"),
            (15, "9"),
        }:
            return 14
        elif key in {
            (0, "9"),
            (0, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "10"),
            (5, "11"),
            (5, "12"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "</s>"),
            (5, "<s>"),
            (14, "7"),
            (14, "<s>"),
        }:
            return 0
        elif key in {
            (1, "10"),
            (1, "11"),
            (1, "12"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
            (13, "3"),
            (13, "5"),
            (13, "9"),
            (14, "5"),
            (15, "5"),
        }:
            return 10
        elif key in {
            (1, "0"),
            (1, "1"),
            (2, "1"),
            (3, "1"),
            (7, "1"),
            (8, "1"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
            (12, "1"),
            (13, "1"),
            (14, "1"),
            (15, "1"),
        }:
            return 5
        elif key in {(14, "12"), (15, "12")}:
            return 12
        elif key in {(15, "2")}:
            return 11
        return 2

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(positions, attn_1_7_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("1", 10),
            ("1", 13),
            ("3", 9),
            ("3", 10),
            ("3", 13),
            ("4", 2),
            ("4", 3),
            ("4", 7),
            ("4", 8),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 15),
            ("5", 2),
            ("5", 3),
            ("5", 10),
            ("5", 13),
            ("6", 2),
            ("6", 3),
            ("7", 2),
            ("7", 3),
            ("7", 10),
            ("7", 13),
            ("8", 2),
            ("8", 3),
            ("8", 7),
            ("8", 8),
            ("8", 10),
            ("8", 11),
            ("8", 12),
            ("8", 13),
            ("8", 14),
            ("8", 15),
            ("9", 2),
            ("9", 3),
            ("9", 10),
            ("9", 13),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 7),
            ("</s>", 8),
            ("</s>", 10),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 15),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 10),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 15),
        }:
            return 10
        elif key in {
            ("0", 9),
            ("1", 9),
            ("10", 9),
            ("11", 9),
            ("12", 9),
            ("2", 9),
            ("4", 4),
            ("4", 5),
            ("4", 9),
            ("5", 0),
            ("5", 4),
            ("5", 5),
            ("5", 6),
            ("5", 7),
            ("5", 8),
            ("5", 9),
            ("5", 11),
            ("5", 12),
            ("5", 14),
            ("5", 15),
            ("6", 9),
            ("6", 15),
            ("7", 4),
            ("7", 5),
            ("7", 7),
            ("7", 8),
            ("7", 9),
            ("7", 11),
            ("7", 12),
            ("7", 14),
            ("7", 15),
            ("8", 4),
            ("8", 5),
            ("8", 9),
            ("9", 4),
            ("9", 5),
            ("9", 7),
            ("9", 8),
            ("9", 9),
            ("9", 11),
            ("9", 12),
            ("9", 14),
            ("9", 15),
            ("</s>", 5),
            ("</s>", 9),
            ("<s>", 5),
            ("<s>", 9),
        }:
            return 13
        elif key in {
            ("0", 1),
            ("0", 12),
            ("1", 1),
            ("12", 1),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("6", 7),
            ("6", 10),
            ("6", 12),
            ("6", 13),
            ("7", 1),
            ("8", 1),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 1
        elif key in {
            ("0", 5),
            ("1", 5),
            ("10", 5),
            ("11", 5),
            ("12", 5),
            ("2", 5),
            ("4", 6),
            ("6", 4),
            ("6", 5),
            ("6", 6),
            ("6", 8),
            ("6", 11),
            ("6", 14),
            ("7", 6),
            ("8", 6),
            ("9", 6),
        }:
            return 15
        elif key in {
            ("0", 2),
            ("0", 10),
            ("1", 2),
            ("11", 2),
            ("12", 2),
            ("3", 2),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 8
        elif key in {("0", 7), ("1", 7), ("12", 7), ("3", 7)}:
            return 6
        elif key in {("10", 1), ("11", 1)}:
            return 0
        return 11

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {
            (5, "0"),
            (5, "1"),
            (5, "10"),
            (5, "11"),
            (5, "12"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "10"),
            (6, "11"),
            (6, "12"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "6"),
            (6, "7"),
            (6, "8"),
            (6, "9"),
            (6, "</s>"),
            (6, "<s>"),
            (14, "6"),
        }:
            return 11
        elif key in {
            (2, "1"),
            (2, "10"),
            (3, "1"),
            (3, "10"),
            (4, "1"),
            (7, "1"),
            (7, "10"),
            (8, "1"),
            (8, "10"),
            (9, "1"),
            (9, "10"),
            (10, "1"),
            (10, "10"),
            (11, "1"),
            (11, "10"),
            (12, "1"),
            (12, "10"),
            (13, "1"),
            (13, "10"),
            (14, "1"),
            (14, "10"),
            (15, "1"),
            (15, "10"),
        }:
            return 14
        elif key in {
            (0, "0"),
            (0, "1"),
            (1, "0"),
            (1, "1"),
            (1, "10"),
            (2, "0"),
            (3, "0"),
            (4, "0"),
            (7, "0"),
            (8, "0"),
            (9, "0"),
            (10, "0"),
            (11, "0"),
            (12, "0"),
            (13, "0"),
            (14, "0"),
            (15, "0"),
        }:
            return 10
        elif key in {
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "6"),
            (7, "2"),
            (7, "4"),
            (8, "2"),
            (8, "4"),
            (8, "6"),
            (9, "4"),
            (10, "2"),
            (10, "4"),
            (11, "4"),
            (12, "2"),
            (12, "4"),
        }:
            return 7
        elif key in {
            (1, "11"),
            (1, "12"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
            (10, "6"),
        }:
            return 1
        elif key in {(0, "10")}:
            return 9
        elif key in {(0, "</s>")}:
            return 13
        return 15

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_5_output, attn_1_1_output):
        key = (attn_1_5_output, attn_1_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "11"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "6"),
            ("0", "7"),
            ("0", "8"),
            ("0", "9"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "10"),
            ("1", "11"),
            ("1", "12"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "6"),
            ("1", "7"),
            ("1", "8"),
            ("1", "9"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "11"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "9"),
            ("10", "</s>"),
            ("10", "<s>"),
            ("11", "0"),
            ("11", "1"),
            ("11", "10"),
            ("11", "11"),
            ("11", "12"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "6"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("11", "</s>"),
            ("11", "<s>"),
            ("12", "10"),
            ("2", "10"),
            ("3", "10"),
            ("4", "10"),
            ("5", "10"),
            ("6", "10"),
            ("7", "10"),
            ("9", "10"),
        }:
            return 2
        elif key in {
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "10"),
            ("</s>", "11"),
            ("</s>", "12"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "5"),
            ("</s>", "6"),
            ("</s>", "7"),
            ("</s>", "8"),
            ("</s>", "9"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "10"),
            ("<s>", "11"),
            ("<s>", "12"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "6"),
            ("<s>", "7"),
            ("<s>", "8"),
            ("<s>", "9"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 6
        elif key in {
            ("8", "10"),
            ("8", "11"),
            ("8", "12"),
            ("8", "2"),
            ("8", "3"),
            ("8", "4"),
            ("8", "5"),
            ("8", "6"),
            ("8", "7"),
            ("8", "8"),
            ("8", "9"),
            ("8", "</s>"),
            ("8", "<s>"),
        }:
            return 10
        return 1

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 12, 13}:
            return token == "6"
        elif mlp_0_0_output in {1, 10, 3}:
            return token == "1"
        elif mlp_0_0_output in {2, 7}:
            return token == "10"
        elif mlp_0_0_output in {4, 5, 6}:
            return token == "<s>"
        elif mlp_0_0_output in {8}:
            return token == "</s>"
        elif mlp_0_0_output in {9}:
            return token == "11"
        elif mlp_0_0_output in {11}:
            return token == "4"
        elif mlp_0_0_output in {14, 15}:
            return token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 2, 3, 8, 12, 13, 14, 15}:
            return token == "</s>"
        elif mlp_0_2_output in {1}:
            return token == "1"
        elif mlp_0_2_output in {4}:
            return token == "2"
        elif mlp_0_2_output in {5}:
            return token == "3"
        elif mlp_0_2_output in {6}:
            return token == "10"
        elif mlp_0_2_output in {7}:
            return token == "7"
        elif mlp_0_2_output in {9}:
            return token == "11"
        elif mlp_0_2_output in {10, 11}:
            return token == "<s>"

    attn_2_1_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 11, 14, 15}:
            return token == "<pad>"
        elif position in {1, 2, 3, 8, 9, 10}:
            return token == "12"
        elif position in {4, 5, 6}:
            return token == "8"
        elif position in {12, 7}:
            return token == "2"
        elif position in {13}:
            return token == "5"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0}:
            return token == "7"
        elif mlp_0_1_output in {1, 10}:
            return token == "1"
        elif mlp_0_1_output in {2}:
            return token == "0"
        elif mlp_0_1_output in {3}:
            return token == "3"
        elif mlp_0_1_output in {9, 4, 13}:
            return token == "6"
        elif mlp_0_1_output in {11, 12, 5}:
            return token == "<s>"
        elif mlp_0_1_output in {6}:
            return token == "4"
        elif mlp_0_1_output in {7}:
            return token == "11"
        elif mlp_0_1_output in {8}:
            return token == "2"
        elif mlp_0_1_output in {14, 15}:
            return token == "<pad>"

    attn_2_3_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_6_output, mlp_1_1_output):
        if attn_0_6_output in {"0", "2"}:
            return mlp_1_1_output == 3
        elif attn_0_6_output in {"1"}:
            return mlp_1_1_output == 1
        elif attn_0_6_output in {"11", "10", "12"}:
            return mlp_1_1_output == 2
        elif attn_0_6_output in {"3"}:
            return mlp_1_1_output == 11
        elif attn_0_6_output in {"4"}:
            return mlp_1_1_output == 12
        elif attn_0_6_output in {"5"}:
            return mlp_1_1_output == 4
        elif attn_0_6_output in {"6"}:
            return mlp_1_1_output == 14
        elif attn_0_6_output in {"7"}:
            return mlp_1_1_output == 13
        elif attn_0_6_output in {"8", "<s>", "9"}:
            return mlp_1_1_output == 0
        elif attn_0_6_output in {"</s>"}:
            return mlp_1_1_output == 15

    attn_2_4_pattern = select_closest(mlp_1_1_outputs, attn_0_6_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(mlp_0_1_output, token):
        if mlp_0_1_output in {0}:
            return token == "12"
        elif mlp_0_1_output in {1, 9}:
            return token == "10"
        elif mlp_0_1_output in {2}:
            return token == "0"
        elif mlp_0_1_output in {8, 10, 3}:
            return token == "1"
        elif mlp_0_1_output in {4}:
            return token == "9"
        elif mlp_0_1_output in {13, 11, 12, 5}:
            return token == "6"
        elif mlp_0_1_output in {15, 6, 14}:
            return token == "<pad>"
        elif mlp_0_1_output in {7}:
            return token == "2"

    attn_2_5_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, tokens)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, token):
        if position in {0, 11, 12, 13, 14, 15}:
            return token == "<pad>"
        elif position in {1, 2}:
            return token == "10"
        elif position in {3, 4, 5}:
            return token == "9"
        elif position in {6}:
            return token == "7"
        elif position in {8, 9, 7}:
            return token == "2"
        elif position in {10}:
            return token == "0"

    attn_2_6_pattern = select_closest(tokens, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 5}:
            return token == "9"
        elif mlp_0_0_output in {1}:
            return token == "1"
        elif mlp_0_0_output in {2, 3}:
            return token == "12"
        elif mlp_0_0_output in {4}:
            return token == "10"
        elif mlp_0_0_output in {6}:
            return token == "<pad>"
        elif mlp_0_0_output in {9, 7}:
            return token == "4"
        elif mlp_0_0_output in {8, 11}:
            return token == "</s>"
        elif mlp_0_0_output in {10}:
            return token == "0"
        elif mlp_0_0_output in {12, 14, 15}:
            return token == "6"
        elif mlp_0_0_output in {13}:
            return token == "8"

    attn_2_7_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_1_output):
        key = (attn_2_3_output, attn_2_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "6"),
            ("0", "7"),
            ("0", "8"),
            ("0", "9"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "10"),
            ("1", "12"),
            ("1", "2"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "6"),
            ("10", "8"),
            ("10", "</s>"),
            ("12", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "12"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "8"),
            ("4", "</s>"),
            ("5", "0"),
            ("5", "10"),
            ("5", "12"),
            ("5", "5"),
            ("5", "6"),
            ("5", "</s>"),
            ("6", "0"),
            ("6", "10"),
            ("6", "12"),
            ("6", "5"),
            ("6", "6"),
            ("6", "</s>"),
            ("7", "0"),
            ("8", "0"),
            ("9", "0"),
            ("</s>", "0"),
            ("<s>", "0"),
            ("<s>", "10"),
            ("<s>", "12"),
            ("<s>", "5"),
            ("<s>", "6"),
            ("<s>", "</s>"),
        }:
            return 4
        elif key in {
            ("1", "11"),
            ("10", "11"),
            ("11", "1"),
            ("11", "10"),
            ("11", "11"),
            ("11", "12"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "6"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("11", "</s>"),
            ("11", "<s>"),
            ("12", "11"),
            ("2", "11"),
            ("3", "11"),
            ("4", "11"),
            ("5", "11"),
            ("6", "11"),
            ("7", "11"),
            ("8", "11"),
            ("9", "11"),
            ("</s>", "11"),
            ("<s>", "11"),
        }:
            return 9
        elif key in {("0", "11"), ("11", "0")}:
            return 0
        return 3

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_4_output, attn_2_1_output):
        key = (attn_2_4_output, attn_2_1_output)
        if key in {
            ("0", "0"),
            ("0", "10"),
            ("0", "12"),
            ("0", "3"),
            ("0", "5"),
            ("0", "7"),
            ("0", "8"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "12"),
            ("10", "0"),
            ("10", "10"),
            ("10", "12"),
            ("10", "3"),
            ("10", "5"),
            ("10", "7"),
            ("10", "8"),
            ("10", "</s>"),
            ("12", "0"),
            ("12", "10"),
            ("12", "12"),
            ("12", "5"),
            ("12", "8"),
            ("12", "</s>"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("5", "10"),
            ("5", "12"),
            ("5", "5"),
            ("5", "7"),
            ("5", "</s>"),
            ("6", "10"),
            ("6", "12"),
            ("6", "<s>"),
            ("8", "0"),
            ("8", "10"),
            ("8", "12"),
            ("8", "5"),
            ("8", "8"),
            ("8", "</s>"),
            ("</s>", "10"),
            ("</s>", "12"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "10"),
            ("<s>", "11"),
            ("<s>", "12"),
            ("<s>", "5"),
            ("<s>", "8"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 11
        elif key in {
            ("0", "11"),
            ("0", "2"),
            ("0", "6"),
            ("1", "2"),
            ("1", "6"),
            ("1", "<s>"),
            ("10", "11"),
            ("10", "2"),
            ("10", "6"),
            ("10", "<s>"),
            ("11", "6"),
            ("12", "11"),
            ("12", "6"),
            ("12", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "12"),
            ("2", "2"),
            ("2", "5"),
            ("2", "6"),
            ("2", "8"),
            ("2", "9"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("5", "11"),
            ("5", "2"),
            ("5", "6"),
            ("5", "<s>"),
            ("6", "2"),
            ("6", "6"),
            ("8", "11"),
            ("8", "2"),
            ("8", "6"),
            ("8", "<s>"),
            ("9", "11"),
            ("9", "2"),
            ("9", "6"),
            ("</s>", "2"),
            ("</s>", "6"),
            ("<s>", "6"),
        }:
            return 13
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("10", "4"),
            ("11", "4"),
            ("12", "2"),
            ("12", "4"),
            ("2", "3"),
            ("2", "4"),
            ("2", "7"),
            ("3", "2"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "12"),
            ("4", "2"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "8"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("5", "4"),
            ("6", "4"),
            ("7", "2"),
            ("7", "4"),
            ("7", "6"),
            ("8", "4"),
            ("</s>", "4"),
            ("<s>", "2"),
            ("<s>", "4"),
        }:
            return 4
        elif key in {("3", "6"), ("4", "3"), ("4", "7"), ("4", "9"), ("9", "4")}:
            return 7
        return 15

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_2_1_output):
        key = (position, attn_2_1_output)
        if key in {
            (5, "1"),
            (5, "10"),
            (5, "11"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "6"),
            (6, "0"),
            (6, "1"),
            (6, "10"),
            (6, "11"),
            (6, "3"),
            (6, "4"),
            (6, "7"),
            (13, "</s>"),
            (14, "3"),
        }:
            return 5
        elif key in {
            (0, "</s>"),
            (0, "<s>"),
            (5, "7"),
            (5, "8"),
            (5, "9"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "12"),
            (6, "8"),
            (6, "9"),
            (6, "</s>"),
            (6, "<s>"),
            (11, "</s>"),
            (12, "</s>"),
            (12, "<s>"),
            (14, "9"),
            (14, "</s>"),
            (14, "<s>"),
            (15, "</s>"),
            (15, "<s>"),
        }:
            return 3
        elif key in {
            (0, "3"),
            (2, "3"),
            (3, "3"),
            (4, "3"),
            (7, "3"),
            (7, "6"),
            (7, "</s>"),
            (7, "<s>"),
            (8, "3"),
            (9, "3"),
            (10, "3"),
            (11, "3"),
            (12, "3"),
            (13, "3"),
            (15, "3"),
        }:
            return 1
        elif key in {(1, "3"), (7, "0"), (7, "1"), (7, "10"), (7, "2"), (7, "4")}:
            return 6
        elif key in {(6, "2"), (6, "5"), (6, "6")}:
            return 7
        elif key in {(7, "11"), (7, "12")}:
            return 0
        return 12

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_2_1_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_3_output, attn_2_1_output):
        key = (attn_2_3_output, attn_2_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "5"),
            ("0", "9"),
            ("0", "</s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "10"),
            ("1", "12"),
            ("1", "3"),
            ("1", "5"),
            ("1", "9"),
            ("1", "</s>"),
            ("11", "0"),
            ("11", "10"),
            ("11", "12"),
            ("11", "3"),
            ("11", "5"),
            ("11", "9"),
            ("11", "</s>"),
            ("12", "5"),
            ("12", "9"),
            ("12", "</s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "10"),
            ("3", "12"),
            ("3", "3"),
            ("3", "5"),
            ("3", "9"),
            ("3", "</s>"),
            ("5", "0"),
            ("5", "10"),
            ("5", "12"),
            ("5", "3"),
            ("5", "5"),
            ("5", "9"),
            ("5", "</s>"),
            ("6", "0"),
            ("9", "0"),
            ("9", "10"),
            ("9", "5"),
            ("9", "9"),
            ("9", "</s>"),
            ("</s>", "0"),
            ("</s>", "10"),
            ("</s>", "12"),
            ("</s>", "3"),
            ("</s>", "5"),
            ("</s>", "9"),
            ("</s>", "</s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "10"),
            ("<s>", "12"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "9"),
            ("<s>", "</s>"),
        }:
            return 1
        elif key in {
            ("0", "8"),
            ("0", "<s>"),
            ("1", "8"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "5"),
            ("10", "8"),
            ("10", "9"),
            ("10", "</s>"),
            ("10", "<s>"),
            ("11", "8"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "2"),
            ("12", "8"),
            ("12", "<s>"),
            ("2", "0"),
            ("2", "10"),
            ("2", "5"),
            ("2", "8"),
            ("2", "</s>"),
            ("3", "8"),
            ("5", "8"),
            ("6", "8"),
            ("8", "0"),
            ("8", "1"),
            ("8", "10"),
            ("8", "11"),
            ("8", "12"),
            ("8", "2"),
            ("8", "3"),
            ("8", "5"),
            ("8", "8"),
            ("8", "9"),
            ("8", "</s>"),
            ("8", "<s>"),
            ("9", "8"),
            ("</s>", "8"),
            ("<s>", "8"),
        }:
            return 13
        elif key in {
            ("0", "6"),
            ("1", "6"),
            ("10", "6"),
            ("11", "6"),
            ("12", "6"),
            ("3", "6"),
            ("4", "6"),
            ("4", "7"),
            ("5", "6"),
            ("6", "10"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "9"),
            ("6", "</s>"),
            ("7", "4"),
            ("7", "6"),
            ("8", "6"),
            ("9", "4"),
            ("9", "6"),
            ("</s>", "6"),
            ("<s>", "6"),
        }:
            return 10
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("11", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "12"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "9"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("5", "4"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 5
        elif key in {("10", "4"), ("2", "4"), ("2", "6"), ("4", "8"), ("8", "4")}:
            return 11
        elif key in {("1", "2"), ("12", "3"), ("3", "2"), ("<s>", "2")}:
            return 8
        elif key in {("12", "4"), ("4", "2")}:
            return 14
        return 9

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                one_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "8", "4", "4", "0", "6", "9", "</s>"]))

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


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/sort_1_8/sort_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(position, token):
        if position in {0, 3, 4, 5, 6, 29}:
            return token == "9"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "0"
        elif position in {7}:
            return token == "15"
        elif position in {8}:
            return token == "14"
        elif position in {9, 27, 21}:
            return token == "3"
        elif position in {10}:
            return token == "16"
        elif position in {11, 20}:
            return token == "23"
        elif position in {12, 15}:
            return token == "18"
        elif position in {13}:
            return token == "27"
        elif position in {18, 14}:
            return token == "28"
        elif position in {16}:
            return token == "24"
        elif position in {17, 19}:
            return token == "22"
        elif position in {22}:
            return token == "6"
        elif position in {25, 28, 23}:
            return token == "8"
        elif position in {24}:
            return token == "5"
        elif position in {26, 31}:
            return token == "7"
        elif position in {30}:
            return token == "25"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 6}:
            return k_position == 5
        elif q_position in {29, 5, 7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 20
        elif q_position in {9, 11, 12}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {21, 19, 13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16, 25}:
            return k_position == 9
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 24
        elif q_position in {20, 28, 31}:
            return k_position == 29
        elif q_position in {24, 22}:
            return k_position == 23
        elif q_position in {26, 23}:
            return k_position == 16
        elif q_position in {27}:
            return k_position == 11
        elif q_position in {30}:
            return k_position == 22

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "13"
        elif position in {2}:
            return token == "10"
        elif position in {3, 4, 5, 6}:
            return token == "8"
        elif position in {18, 15, 7}:
            return token == "18"
        elif position in {8, 27}:
            return token == "3"
        elif position in {9, 19}:
            return token == "24"
        elif position in {10}:
            return token == "21"
        elif position in {11}:
            return token == "14"
        elif position in {12}:
            return token == "16"
        elif position in {13}:
            return token == "27"
        elif position in {14, 30}:
            return token == "19"
        elif position in {16}:
            return token == "12"
        elif position in {17, 21}:
            return token == "17"
        elif position in {20}:
            return token == "22"
        elif position in {24, 22}:
            return token == "9"
        elif position in {23}:
            return token == "0"
        elif position in {25}:
            return token == "23"
        elif position in {26, 29}:
            return token == "4"
        elif position in {28}:
            return token == "5"
        elif position in {31}:
            return token == "20"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 31}:
            return k_position == 3
        elif q_position in {1, 5}:
            return k_position == 4
        elif q_position in {2, 3}:
            return k_position == 1
        elif q_position in {4, 29}:
            return k_position == 25
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {27, 7}:
            return k_position == 6
        elif q_position in {8, 11, 12}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 9
        elif q_position in {16, 15}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {18, 20}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {24, 23}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {28}:
            return k_position == 19
        elif q_position in {30}:
            return k_position == 27

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 2, 11, 16, 27, 30, 31}:
            return k_position == 6
        elif q_position in {1, 21}:
            return k_position == 3
        elif q_position in {3, 4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 14
        elif q_position in {12, 6}:
            return k_position == 28
        elif q_position in {17, 18, 22, 7}:
            return k_position == 5
        elif q_position in {8, 9, 13, 14, 15, 20, 24, 25}:
            return k_position == 16
        elif q_position in {10}:
            return k_position == 21
        elif q_position in {26, 19, 29}:
            return k_position == 12
        elif q_position in {23}:
            return k_position == 2
        elif q_position in {28}:
            return k_position == 1

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {
            0,
            4,
            7,
            8,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            27,
            28,
            30,
            31,
        }:
            return k_position == 6
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {5, 9, 10, 25, 26, 29}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 4, 8, 21, 30, 31}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 3, 7, 18, 20, 26}:
            return k_position == 5
        elif q_position in {
            5,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
        }:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 17

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 7, 8, 9, 10, 12, 19, 28, 30}:
            return k_position == 6
        elif q_position in {
            4,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            31,
        }:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 26
        elif q_position in {6}:
            return k_position == 19

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {2, 9, 11, 13, 14, 16, 17, 19, 22, 23, 26, 27}:
            return 19
        elif key in {1, 15}:
            return 3
        return 17

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 0),
            ("12", 0),
            ("12", 3),
            ("12", 4),
            ("12", 5),
            ("12", 8),
            ("12", 12),
            ("12", 14),
            ("12", 19),
            ("12", 21),
            ("12", 23),
            ("12", 27),
            ("12", 28),
            ("12", 29),
            ("15", 0),
            ("15", 3),
            ("15", 4),
            ("15", 5),
            ("15", 6),
            ("15", 7),
            ("15", 8),
            ("15", 10),
            ("15", 11),
            ("15", 12),
            ("15", 13),
            ("15", 14),
            ("15", 16),
            ("15", 18),
            ("15", 19),
            ("15", 21),
            ("15", 23),
            ("15", 26),
            ("15", 27),
            ("15", 28),
            ("15", 29),
            ("15", 30),
            ("9", 8),
            ("</s>", 0),
            ("</s>", 5),
            ("</s>", 28),
            ("</s>", 29),
        }:
            return 11
        elif key in {
            ("12", 7),
            ("2", 0),
            ("20", 0),
            ("20", 5),
            ("20", 7),
            ("20", 26),
            ("20", 27),
            ("20", 28),
            ("23", 0),
            ("3", 0),
            ("4", 0),
            ("6", 0),
            ("7", 0),
            ("7", 5),
            ("7", 7),
            ("7", 26),
            ("7", 28),
            ("8", 0),
            ("8", 5),
            ("9", 0),
            ("9", 5),
            ("9", 28),
        }:
            return 15
        elif key in {("0", 5), ("0", 7), ("0", 28)}:
            return 10
        return 7

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 15

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, one):
        key = (num_attn_0_1_output, one)
        return 16

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(num_attn_0_1_outputs, ones)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 17, 24}:
            return token == "6"
        elif mlp_0_0_output in {8, 1, 31}:
            return token == "10"
        elif mlp_0_0_output in {2}:
            return token == "14"
        elif mlp_0_0_output in {3}:
            return token == "18"
        elif mlp_0_0_output in {4, 22}:
            return token == "2"
        elif mlp_0_0_output in {26, 5, 6}:
            return token == "9"
        elif mlp_0_0_output in {12, 14, 7}:
            return token == "16"
        elif mlp_0_0_output in {9, 11, 15}:
            return token == "12"
        elif mlp_0_0_output in {10}:
            return token == "15"
        elif mlp_0_0_output in {13}:
            return token == "20"
        elif mlp_0_0_output in {16, 18}:
            return token == "23"
        elif mlp_0_0_output in {19}:
            return token == "17"
        elif mlp_0_0_output in {20}:
            return token == "21"
        elif mlp_0_0_output in {21}:
            return token == "4"
        elif mlp_0_0_output in {30, 23}:
            return token == "27"
        elif mlp_0_0_output in {25, 28}:
            return token == "7"
        elif mlp_0_0_output in {27}:
            return token == "8"
        elif mlp_0_0_output in {29}:
            return token == "3"

    attn_1_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 30}:
            return token == "3"
        elif position in {1}:
            return token == "11"
        elif position in {2}:
            return token == "13"
        elif position in {3}:
            return token == "15"
        elif position in {29, 4, 5}:
            return token == "4"
        elif position in {6}:
            return token == "5"
        elif position in {15, 7}:
            return token == "26"
        elif position in {8, 18, 12}:
            return token == "17"
        elif position in {9}:
            return token == "1"
        elif position in {10}:
            return token == "19"
        elif position in {11}:
            return token == "2"
        elif position in {26, 13}:
            return token == "21"
        elif position in {19, 14}:
            return token == "18"
        elif position in {16}:
            return token == "20"
        elif position in {17}:
            return token == "28"
        elif position in {20}:
            return token == "24"
        elif position in {27, 21}:
            return token == "8"
        elif position in {24, 22}:
            return token == "6"
        elif position in {31, 23}:
            return token == "27"
        elif position in {25}:
            return token == "7"
        elif position in {28}:
            return token == "9"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"0", "20"}:
            return k_token == "16"
        elif q_token in {"</s>", "1"}:
            return k_token == "12"
        elif q_token in {"10"}:
            return k_token == "9"
        elif q_token in {"11", "28", "5"}:
            return k_token == "4"
        elif q_token in {"12", "6"}:
            return k_token == "7"
        elif q_token in {"<s>", "13"}:
            return k_token == "8"
        elif q_token in {"14"}:
            return k_token == "19"
        elif q_token in {"15"}:
            return k_token == "14"
        elif q_token in {"16"}:
            return k_token == "17"
        elif q_token in {"17"}:
            return k_token == "15"
        elif q_token in {"25", "18"}:
            return k_token == "0"
        elif q_token in {"19", "2", "23"}:
            return k_token == "18"
        elif q_token in {"21"}:
            return k_token == "5"
        elif q_token in {"22"}:
            return k_token == "10"
        elif q_token in {"24"}:
            return k_token == "21"
        elif q_token in {"27", "9", "26"}:
            return k_token == "23"
        elif q_token in {"3"}:
            return k_token == "28"
        elif q_token in {"7", "4"}:
            return k_token == "6"
        elif q_token in {"8"}:
            return k_token == "22"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 28}:
            return token == "9"
        elif position in {1}:
            return token == "14"
        elif position in {2}:
            return token == "26"
        elif position in {3, 21, 23}:
            return token == "7"
        elif position in {27, 4, 5, 30}:
            return token == "5"
        elif position in {6}:
            return token == "4"
        elif position in {7}:
            return token == "15"
        elif position in {8}:
            return token == "12"
        elif position in {9}:
            return token == "11"
        elif position in {10}:
            return token == "0"
        elif position in {11, 12}:
            return token == "17"
        elif position in {13}:
            return token == "19"
        elif position in {16, 14}:
            return token == "21"
        elif position in {15}:
            return token == "10"
        elif position in {17, 20}:
            return token == "24"
        elif position in {18, 22}:
            return token == "27"
        elif position in {19}:
            return token == "23"
        elif position in {24, 26}:
            return token == "8"
        elif position in {25, 29, 31}:
            return token == "6"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, mlp_0_0_output):
        if position in {0, 1, 2, 3, 8, 19, 22}:
            return mlp_0_0_output == 17
        elif position in {4, 11, 13, 14, 17, 20, 23, 24, 25, 26, 27, 29, 30}:
            return mlp_0_0_output == 7
        elif position in {5}:
            return mlp_0_0_output == 15
        elif position in {6}:
            return mlp_0_0_output == 16
        elif position in {12, 7}:
            return mlp_0_0_output == 3
        elif position in {16, 9, 31, 15}:
            return mlp_0_0_output == 5
        elif position in {10, 18, 28, 21}:
            return mlp_0_0_output == 6

    num_attn_1_0_pattern = select(mlp_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 12, 24, 25, 28, 30, 31}:
            return token == "11"
        elif position in {1, 23, 14, 22}:
            return token == "10"
        elif position in {2, 20}:
            return token == "0"
        elif position in {3, 4, 5, 9, 15, 17, 26, 27, 29}:
            return token == "19"
        elif position in {6}:
            return token == "<pad>"
        elif position in {18, 7}:
            return token == "23"
        elif position in {8}:
            return token == "20"
        elif position in {16, 10}:
            return token == "13"
        elif position in {21, 11, 13}:
            return token == "25"
        elif position in {19}:
            return token == "1"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 8, 13, 19, 21, 23, 24, 25, 26, 28, 30, 31}:
            return token == "12"
        elif position in {1, 9, 12, 14, 15, 16, 20, 22}:
            return token == "0"
        elif position in {2, 3, 4}:
            return token == "13"
        elif position in {5, 6}:
            return token == "7"
        elif position in {27, 7}:
            return token == "11"
        elif position in {10}:
            return token == "1"
        elif position in {18, 11}:
            return token == "10"
        elif position in {17}:
            return token == "15"
        elif position in {29}:
            return token == "19"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_position, k_position):
        if q_position in {0, 3, 9, 14, 19, 24}:
            return k_position == 5
        elif q_position in {1, 10}:
            return k_position == 2
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {
            5,
            8,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        }:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 18

    num_attn_1_3_pattern = select(positions, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_2_output):
        key = (attn_1_1_output, attn_1_2_output)
        if key in {
            ("0", "10"),
            ("0", "13"),
            ("1", "10"),
            ("1", "13"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "11"),
            ("10", "12"),
            ("10", "13"),
            ("10", "15"),
            ("10", "16"),
            ("10", "17"),
            ("10", "18"),
            ("10", "19"),
            ("10", "2"),
            ("10", "20"),
            ("10", "21"),
            ("10", "22"),
            ("10", "23"),
            ("10", "24"),
            ("10", "25"),
            ("10", "26"),
            ("10", "27"),
            ("10", "28"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "9"),
            ("10", "</s>"),
            ("10", "<s>"),
            ("11", "10"),
            ("11", "13"),
            ("12", "10"),
            ("12", "13"),
            ("12", "6"),
            ("13", "0"),
            ("13", "1"),
            ("13", "10"),
            ("13", "11"),
            ("13", "12"),
            ("13", "13"),
            ("13", "15"),
            ("13", "16"),
            ("13", "17"),
            ("13", "18"),
            ("13", "19"),
            ("13", "2"),
            ("13", "20"),
            ("13", "21"),
            ("13", "22"),
            ("13", "23"),
            ("13", "24"),
            ("13", "25"),
            ("13", "26"),
            ("13", "27"),
            ("13", "28"),
            ("13", "3"),
            ("13", "4"),
            ("13", "5"),
            ("13", "6"),
            ("13", "7"),
            ("13", "8"),
            ("13", "9"),
            ("13", "</s>"),
            ("13", "<s>"),
            ("15", "10"),
            ("15", "13"),
            ("16", "10"),
            ("16", "13"),
            ("17", "10"),
            ("17", "13"),
            ("18", "10"),
            ("18", "13"),
            ("19", "10"),
            ("19", "13"),
            ("2", "10"),
            ("2", "13"),
            ("20", "10"),
            ("20", "13"),
            ("21", "10"),
            ("21", "13"),
            ("22", "10"),
            ("22", "13"),
            ("23", "10"),
            ("23", "13"),
            ("24", "10"),
            ("24", "13"),
            ("25", "10"),
            ("25", "13"),
            ("26", "10"),
            ("26", "13"),
            ("27", "10"),
            ("27", "13"),
            ("28", "0"),
            ("28", "10"),
            ("28", "13"),
            ("28", "18"),
            ("28", "20"),
            ("28", "21"),
            ("28", "22"),
            ("28", "23"),
            ("28", "24"),
            ("28", "26"),
            ("28", "28"),
            ("28", "5"),
            ("28", "6"),
            ("28", "7"),
            ("28", "9"),
            ("28", "</s>"),
            ("3", "10"),
            ("3", "13"),
            ("4", "10"),
            ("4", "13"),
            ("5", "10"),
            ("5", "13"),
            ("6", "1"),
            ("6", "10"),
            ("6", "12"),
            ("6", "13"),
            ("6", "17"),
            ("6", "2"),
            ("6", "20"),
            ("6", "23"),
            ("6", "24"),
            ("6", "28"),
            ("6", "5"),
            ("6", "6"),
            ("6", "9"),
            ("6", "</s>"),
            ("7", "0"),
            ("7", "1"),
            ("7", "10"),
            ("7", "13"),
            ("7", "18"),
            ("7", "20"),
            ("7", "21"),
            ("7", "23"),
            ("7", "24"),
            ("7", "25"),
            ("7", "26"),
            ("7", "5"),
            ("7", "7"),
            ("7", "9"),
            ("7", "</s>"),
            ("8", "10"),
            ("8", "13"),
            ("9", "10"),
            ("9", "13"),
            ("</s>", "10"),
            ("</s>", "13"),
            ("<s>", "10"),
            ("<s>", "13"),
        }:
            return 1
        elif key in {
            ("0", "15"),
            ("0", "25"),
            ("1", "15"),
            ("1", "25"),
            ("11", "1"),
            ("11", "15"),
            ("11", "16"),
            ("12", "0"),
            ("12", "1"),
            ("12", "15"),
            ("12", "25"),
            ("12", "26"),
            ("12", "</s>"),
            ("15", "0"),
            ("15", "1"),
            ("15", "11"),
            ("15", "12"),
            ("15", "15"),
            ("15", "16"),
            ("15", "17"),
            ("15", "18"),
            ("15", "19"),
            ("15", "2"),
            ("15", "20"),
            ("15", "21"),
            ("15", "22"),
            ("15", "23"),
            ("15", "24"),
            ("15", "25"),
            ("15", "26"),
            ("15", "27"),
            ("15", "28"),
            ("15", "3"),
            ("15", "5"),
            ("15", "6"),
            ("15", "7"),
            ("15", "8"),
            ("15", "9"),
            ("15", "</s>"),
            ("15", "<s>"),
            ("16", "15"),
            ("17", "15"),
            ("17", "25"),
            ("19", "15"),
            ("2", "15"),
            ("20", "0"),
            ("20", "15"),
            ("20", "23"),
            ("20", "25"),
            ("20", "26"),
            ("21", "15"),
            ("21", "25"),
            ("22", "15"),
            ("22", "25"),
            ("22", "26"),
            ("23", "15"),
            ("23", "23"),
            ("23", "24"),
            ("23", "25"),
            ("23", "26"),
            ("24", "15"),
            ("24", "25"),
            ("24", "26"),
            ("25", "0"),
            ("25", "1"),
            ("25", "15"),
            ("25", "17"),
            ("25", "23"),
            ("25", "24"),
            ("25", "25"),
            ("25", "26"),
            ("25", "6"),
            ("25", "</s>"),
            ("26", "15"),
            ("26", "25"),
            ("26", "26"),
            ("28", "15"),
            ("28", "25"),
            ("3", "15"),
            ("5", "15"),
            ("6", "15"),
            ("6", "25"),
            ("7", "15"),
            ("8", "15"),
            ("</s>", "15"),
            ("</s>", "25"),
            ("<s>", "15"),
        }:
            return 11
        elif key in {
            ("0", "14"),
            ("1", "14"),
            ("11", "14"),
            ("12", "14"),
            ("14", "0"),
            ("14", "1"),
            ("14", "11"),
            ("14", "12"),
            ("14", "14"),
            ("14", "15"),
            ("14", "16"),
            ("14", "17"),
            ("14", "18"),
            ("14", "19"),
            ("14", "2"),
            ("14", "20"),
            ("14", "21"),
            ("14", "22"),
            ("14", "23"),
            ("14", "24"),
            ("14", "25"),
            ("14", "26"),
            ("14", "27"),
            ("14", "28"),
            ("14", "3"),
            ("14", "4"),
            ("14", "5"),
            ("14", "6"),
            ("14", "7"),
            ("14", "8"),
            ("14", "9"),
            ("14", "</s>"),
            ("14", "<s>"),
            ("15", "14"),
            ("16", "14"),
            ("17", "14"),
            ("18", "14"),
            ("19", "14"),
            ("2", "14"),
            ("20", "14"),
            ("21", "14"),
            ("22", "14"),
            ("23", "14"),
            ("24", "14"),
            ("25", "14"),
            ("26", "14"),
            ("27", "14"),
            ("28", "14"),
            ("3", "14"),
            ("4", "14"),
            ("5", "14"),
            ("5", "5"),
            ("6", "14"),
            ("7", "14"),
            ("8", "14"),
            ("9", "14"),
            ("</s>", "14"),
            ("<s>", "14"),
        }:
            return 13
        elif key in {("17", "0"), ("23", "0"), ("6", "0"), ("6", "26")}:
            return 24
        elif key in {("10", "14"), ("13", "14"), ("14", "10"), ("14", "13")}:
            return 31
        return 27

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, mlp_0_0_output):
        key = (attn_1_1_output, mlp_0_0_output)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 9),
            ("0", 11),
            ("0", 13),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 19),
            ("0", 20),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("0", 30),
            ("1", 1),
            ("1", 2),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 9),
            ("1", 11),
            ("1", 13),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 19),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 30),
            ("10", 1),
            ("10", 2),
            ("10", 3),
            ("10", 4),
            ("10", 5),
            ("10", 6),
            ("10", 7),
            ("10", 9),
            ("10", 11),
            ("10", 13),
            ("10", 14),
            ("10", 15),
            ("10", 16),
            ("10", 19),
            ("10", 20),
            ("10", 21),
            ("10", 22),
            ("10", 26),
            ("10", 30),
            ("11", 1),
            ("11", 2),
            ("11", 4),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 9),
            ("11", 11),
            ("11", 13),
            ("11", 14),
            ("11", 15),
            ("11", 16),
            ("11", 17),
            ("11", 19),
            ("11", 20),
            ("11", 21),
            ("11", 22),
            ("11", 23),
            ("11", 25),
            ("11", 26),
            ("11", 27),
            ("11", 28),
            ("11", 29),
            ("11", 30),
            ("11", 31),
            ("12", 1),
            ("12", 2),
            ("12", 3),
            ("12", 4),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("12", 9),
            ("12", 11),
            ("12", 13),
            ("12", 14),
            ("12", 15),
            ("12", 16),
            ("12", 17),
            ("12", 19),
            ("12", 20),
            ("12", 21),
            ("12", 22),
            ("12", 23),
            ("12", 24),
            ("12", 25),
            ("12", 26),
            ("12", 27),
            ("12", 28),
            ("12", 29),
            ("12", 30),
            ("12", 31),
            ("13", 1),
            ("13", 5),
            ("16", 1),
            ("16", 3),
            ("16", 4),
            ("16", 5),
            ("18", 3),
            ("18", 5),
            ("2", 3),
            ("2", 5),
            ("20", 1),
            ("20", 3),
            ("20", 4),
            ("20", 5),
            ("21", 1),
            ("21", 3),
            ("21", 4),
            ("21", 5),
            ("22", 1),
            ("22", 3),
            ("22", 4),
            ("22", 5),
            ("23", 3),
            ("25", 1),
            ("25", 2),
            ("25", 3),
            ("25", 4),
            ("25", 5),
            ("26", 1),
            ("26", 2),
            ("26", 3),
            ("26", 4),
            ("26", 5),
            ("27", 3),
            ("27", 4),
            ("27", 5),
            ("28", 1),
            ("28", 3),
            ("28", 4),
            ("28", 5),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("4", 1),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("5", 3),
            ("5", 4),
            ("5", 5),
            ("6", 1),
            ("6", 2),
            ("6", 3),
            ("6", 4),
            ("6", 5),
            ("6", 13),
            ("6", 15),
            ("6", 20),
            ("7", 1),
            ("7", 2),
            ("7", 3),
            ("7", 4),
            ("7", 5),
            ("8", 3),
            ("8", 5),
            ("9", 1),
            ("9", 2),
            ("9", 3),
            ("9", 4),
            ("9", 5),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 4
        elif key in {
            ("0", 10),
            ("0", 18),
            ("0", 31),
            ("1", 10),
            ("1", 18),
            ("10", 10),
            ("10", 18),
            ("11", 10),
            ("11", 18),
            ("12", 10),
            ("12", 18),
            ("13", 4),
            ("13", 10),
            ("13", 18),
            ("14", 0),
            ("14", 1),
            ("14", 2),
            ("14", 3),
            ("14", 4),
            ("14", 5),
            ("14", 6),
            ("14", 7),
            ("14", 8),
            ("14", 9),
            ("14", 10),
            ("14", 11),
            ("14", 12),
            ("14", 13),
            ("14", 14),
            ("14", 15),
            ("14", 16),
            ("14", 17),
            ("14", 18),
            ("14", 19),
            ("14", 20),
            ("14", 21),
            ("14", 22),
            ("14", 23),
            ("14", 24),
            ("14", 25),
            ("14", 26),
            ("14", 27),
            ("14", 28),
            ("14", 29),
            ("14", 30),
            ("14", 31),
            ("15", 1),
            ("15", 4),
            ("15", 10),
            ("15", 18),
            ("16", 10),
            ("16", 18),
            ("17", 0),
            ("17", 1),
            ("17", 2),
            ("17", 3),
            ("17", 4),
            ("17", 5),
            ("17", 6),
            ("17", 7),
            ("17", 8),
            ("17", 9),
            ("17", 10),
            ("17", 11),
            ("17", 12),
            ("17", 13),
            ("17", 14),
            ("17", 15),
            ("17", 16),
            ("17", 17),
            ("17", 18),
            ("17", 19),
            ("17", 20),
            ("17", 21),
            ("17", 22),
            ("17", 23),
            ("17", 24),
            ("17", 25),
            ("17", 26),
            ("17", 27),
            ("17", 28),
            ("17", 29),
            ("17", 30),
            ("17", 31),
            ("18", 10),
            ("18", 18),
            ("19", 1),
            ("19", 4),
            ("19", 5),
            ("19", 10),
            ("19", 18),
            ("2", 10),
            ("2", 18),
            ("20", 10),
            ("20", 18),
            ("21", 10),
            ("21", 18),
            ("22", 10),
            ("22", 18),
            ("23", 4),
            ("23", 10),
            ("23", 18),
            ("24", 0),
            ("24", 1),
            ("24", 2),
            ("24", 3),
            ("24", 4),
            ("24", 5),
            ("24", 6),
            ("24", 7),
            ("24", 8),
            ("24", 9),
            ("24", 10),
            ("24", 11),
            ("24", 12),
            ("24", 13),
            ("24", 14),
            ("24", 15),
            ("24", 16),
            ("24", 17),
            ("24", 18),
            ("24", 19),
            ("24", 20),
            ("24", 21),
            ("24", 22),
            ("24", 23),
            ("24", 24),
            ("24", 25),
            ("24", 26),
            ("24", 27),
            ("24", 28),
            ("24", 29),
            ("24", 30),
            ("24", 31),
            ("25", 10),
            ("25", 18),
            ("26", 10),
            ("26", 18),
            ("27", 10),
            ("27", 18),
            ("28", 10),
            ("28", 18),
            ("3", 10),
            ("3", 18),
            ("4", 10),
            ("4", 18),
            ("5", 10),
            ("5", 18),
            ("6", 10),
            ("6", 18),
            ("7", 10),
            ("7", 18),
            ("8", 4),
            ("8", 10),
            ("8", 18),
            ("9", 10),
            ("9", 18),
            ("</s>", 10),
            ("</s>", 18),
            ("<s>", 10),
            ("<s>", 18),
        }:
            return 29
        elif key in {("0", 12), ("1", 8), ("11", 8), ("12", 8), ("12", 12), ("15", 5)}:
            return 22
        elif key in {("1", 3), ("11", 3), ("13", 3)}:
            return 5
        elif key in {("0", 8), ("10", 8), ("19", 3)}:
            return 11
        elif key in {("15", 3)}:
            return 8
        return 23

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        return 21

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_0_3_output):
        key = (num_attn_1_1_output, num_attn_0_3_output)
        return 25

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 4}:
            return k_position == 7
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 30
        elif q_position in {6}:
            return k_position == 25
        elif q_position in {15, 7}:
            return k_position == 20
        elif q_position in {8, 12}:
            return k_position == 13
        elif q_position in {9, 28, 21}:
            return k_position == 4
        elif q_position in {10, 19, 22}:
            return k_position == 12
        elif q_position in {18, 11}:
            return k_position == 16
        elif q_position in {13, 20, 23, 25, 29}:
            return k_position == 21
        elif q_position in {16, 14}:
            return k_position == 17
        elif q_position in {17, 26, 30, 31}:
            return k_position == 22
        elif q_position in {24, 27}:
            return k_position == 23

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 7, 20, 24, 27, 30}:
            return k_position == 21
        elif q_position in {1, 13, 6}:
            return k_position == 3
        elif q_position in {2, 8, 10, 14, 18, 22}:
            return k_position == 4
        elif q_position in {3, 4, 5, 15}:
            return k_position == 2
        elif q_position in {9, 12}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {16, 31}:
            return k_position == 18
        elif q_position in {17, 21}:
            return k_position == 27
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {26, 23}:
            return k_position == 29
        elif q_position in {25}:
            return k_position == 6
        elif q_position in {28}:
            return k_position == 5
        elif q_position in {29}:
            return k_position == 11

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 12}:
            return token == "4"
        elif position in {1, 20, 28}:
            return token == "11"
        elif position in {2, 3}:
            return token == "</s>"
        elif position in {4, 5}:
            return token == "<s>"
        elif position in {6, 7}:
            return token == "7"
        elif position in {8, 14}:
            return token == "16"
        elif position in {9, 19, 23, 24, 29}:
            return token == "27"
        elif position in {16, 10}:
            return token == "21"
        elif position in {11}:
            return token == "2"
        elif position in {13, 22}:
            return token == "0"
        elif position in {15, 21, 25, 26, 30, 31}:
            return token == "26"
        elif position in {17}:
            return token == "9"
        elif position in {18}:
            return token == "20"
        elif position in {27}:
            return token == "6"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 9, 30}:
            return k_position == 29
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 1
        elif q_position in {25, 5}:
            return k_position == 23
        elif q_position in {6}:
            return k_position == 22
        elif q_position in {16, 19, 7}:
            return k_position == 18
        elif q_position in {8}:
            return k_position == 14
        elif q_position in {10, 26, 15}:
            return k_position == 20
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {17, 27, 13}:
            return k_position == 26
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {18, 31}:
            return k_position == 24
        elif q_position in {20, 29, 23}:
            return k_position == 21
        elif q_position in {24, 28, 21, 22}:
            return k_position == 25

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
            7,
            8,
            9,
            10,
            11,
            15,
            20,
            22,
            23,
            24,
            25,
            29,
            30,
            31,
        }:
            return token == "12"
        elif mlp_0_0_output in {4, 5, 14, 18, 21, 28}:
            return token == "10"
        elif mlp_0_0_output in {6}:
            return token == "<pad>"
        elif mlp_0_0_output in {16, 27, 12, 13}:
            return token == "11"
        elif mlp_0_0_output in {17}:
            return token == "0"
        elif mlp_0_0_output in {19}:
            return token == "1"
        elif mlp_0_0_output in {26}:
            return token == "19"

    num_attn_2_0_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 7, 9, 30, 31}:
            return token == "12"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4}:
            return token == "1"
        elif position in {5, 6}:
            return token == "<pad>"
        elif position in {8, 10, 12, 13, 14, 15, 16, 20, 22, 23, 24, 26, 27, 28, 29}:
            return token == "11"
        elif position in {11, 18, 19, 21, 25}:
            return token == "10"
        elif position in {17}:
            return token == "0"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_position, k_position):
        if q_position in {0, 4, 11, 17, 20, 21, 23, 24, 26, 27, 29}:
            return k_position == 7
        elif q_position in {8, 1, 9, 7}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 10, 12, 13, 14, 15, 16, 19, 22, 30, 31}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {25}:
            return k_position == 6
        elif q_position in {28}:
            return k_position == 5

    num_attn_2_2_pattern = select(positions, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, token):
        if attn_1_1_output in {
            "8",
            "18",
            "21",
            "17",
            "5",
            "<s>",
            "23",
            "25",
            "19",
            "28",
            "1",
            "26",
            "13",
            "</s>",
            "24",
            "22",
            "3",
            "9",
            "14",
            "15",
            "0",
            "27",
            "4",
            "20",
            "6",
            "7",
            "2",
            "12",
        }:
            return token == "11"
        elif attn_1_1_output in {"10"}:
            return token == "10"
        elif attn_1_1_output in {"11"}:
            return token == "</s>"
        elif attn_1_1_output in {"16"}:
            return token == "16"

    num_attn_2_3_pattern = select(tokens, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, attn_2_2_output):
        key = (position, attn_2_2_output)
        if key in {
            (0, "0"),
            (2, "0"),
            (3, "0"),
            (4, "0"),
            (7, "0"),
            (7, "10"),
            (8, "0"),
            (8, "1"),
            (8, "10"),
            (9, "0"),
            (9, "1"),
            (9, "10"),
            (10, "0"),
            (10, "10"),
            (11, "0"),
            (12, "0"),
            (13, "0"),
            (14, "0"),
            (15, "0"),
            (16, "0"),
            (18, "0"),
            (19, "0"),
            (20, "0"),
            (21, "0"),
            (22, "0"),
            (23, "0"),
            (24, "0"),
            (25, "0"),
            (26, "0"),
            (27, "0"),
            (28, "0"),
            (29, "0"),
            (30, "0"),
            (31, "0"),
        }:
            return 17
        elif key in {(1, "0"), (1, "1"), (1, "10"), (1, "12")}:
            return 6
        return 7

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(positions, attn_2_2_outputs)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, attn_2_0_output):
        key = (attn_2_3_output, attn_2_0_output)
        if key in {
            ("0", "10"),
            ("0", "11"),
            ("0", "24"),
            ("0", "25"),
            ("1", "10"),
            ("1", "11"),
            ("1", "25"),
            ("10", "10"),
            ("10", "11"),
            ("10", "16"),
            ("10", "24"),
            ("10", "25"),
            ("10", "</s>"),
            ("11", "10"),
            ("11", "11"),
            ("11", "25"),
            ("14", "11"),
            ("15", "11"),
            ("16", "11"),
            ("17", "10"),
            ("17", "11"),
            ("17", "25"),
            ("18", "11"),
            ("18", "25"),
            ("19", "10"),
            ("19", "11"),
            ("19", "25"),
            ("19", "</s>"),
            ("2", "10"),
            ("2", "11"),
            ("2", "24"),
            ("2", "25"),
            ("20", "11"),
            ("22", "10"),
            ("22", "11"),
            ("22", "18"),
            ("22", "25"),
            ("22", "</s>"),
            ("23", "10"),
            ("23", "11"),
            ("23", "25"),
            ("25", "10"),
            ("25", "11"),
            ("25", "25"),
            ("26", "10"),
            ("26", "11"),
            ("26", "25"),
            ("27", "0"),
            ("27", "1"),
            ("27", "10"),
            ("27", "11"),
            ("27", "14"),
            ("27", "16"),
            ("27", "17"),
            ("27", "18"),
            ("27", "2"),
            ("27", "21"),
            ("27", "24"),
            ("27", "25"),
            ("27", "27"),
            ("27", "4"),
            ("27", "9"),
            ("27", "</s>"),
            ("28", "11"),
            ("4", "10"),
            ("4", "11"),
            ("4", "25"),
            ("5", "11"),
            ("6", "10"),
            ("6", "11"),
            ("6", "18"),
            ("6", "21"),
            ("6", "25"),
            ("6", "</s>"),
            ("7", "10"),
            ("7", "11"),
            ("7", "24"),
            ("7", "25"),
            ("8", "10"),
            ("8", "11"),
            ("8", "14"),
            ("8", "24"),
            ("8", "25"),
            ("8", "</s>"),
            ("9", "10"),
            ("9", "11"),
            ("9", "25"),
            ("9", "</s>"),
            ("</s>", "10"),
            ("</s>", "11"),
            ("</s>", "25"),
            ("<s>", "10"),
            ("<s>", "11"),
            ("<s>", "25"),
        }:
            return 25
        elif key in {
            ("0", "21"),
            ("1", "21"),
            ("10", "21"),
            ("11", "21"),
            ("14", "10"),
            ("14", "18"),
            ("14", "21"),
            ("14", "25"),
            ("14", "</s>"),
            ("16", "21"),
            ("17", "21"),
            ("18", "10"),
            ("18", "14"),
            ("18", "17"),
            ("18", "18"),
            ("18", "20"),
            ("18", "21"),
            ("18", "24"),
            ("18", "26"),
            ("18", "4"),
            ("19", "17"),
            ("19", "18"),
            ("19", "21"),
            ("19", "26"),
            ("19", "4"),
            ("2", "0"),
            ("2", "14"),
            ("2", "18"),
            ("2", "21"),
            ("20", "10"),
            ("20", "18"),
            ("20", "21"),
            ("21", "0"),
            ("21", "10"),
            ("21", "17"),
            ("21", "18"),
            ("21", "20"),
            ("21", "21"),
            ("21", "24"),
            ("21", "25"),
            ("21", "26"),
            ("21", "4"),
            ("21", "<s>"),
            ("22", "21"),
            ("23", "21"),
            ("25", "18"),
            ("25", "21"),
            ("26", "18"),
            ("26", "21"),
            ("28", "21"),
            ("4", "18"),
            ("4", "21"),
            ("5", "21"),
            ("8", "21"),
            ("9", "17"),
            ("9", "18"),
            ("9", "20"),
            ("9", "21"),
            ("9", "26"),
            ("</s>", "21"),
            ("<s>", "18"),
            ("<s>", "21"),
        }:
            return 30
        elif key in {
            ("0", "12"),
            ("1", "12"),
            ("10", "12"),
            ("11", "12"),
            ("12", "0"),
            ("12", "1"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "13"),
            ("12", "14"),
            ("12", "15"),
            ("12", "17"),
            ("12", "18"),
            ("12", "19"),
            ("12", "2"),
            ("12", "20"),
            ("12", "21"),
            ("12", "22"),
            ("12", "23"),
            ("12", "24"),
            ("12", "25"),
            ("12", "26"),
            ("12", "27"),
            ("12", "28"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "6"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("12", "</s>"),
            ("12", "<s>"),
            ("13", "12"),
            ("14", "12"),
            ("15", "12"),
            ("16", "12"),
            ("17", "12"),
            ("18", "12"),
            ("19", "12"),
            ("2", "12"),
            ("20", "12"),
            ("21", "12"),
            ("22", "12"),
            ("23", "12"),
            ("24", "12"),
            ("25", "12"),
            ("26", "12"),
            ("27", "12"),
            ("28", "12"),
            ("3", "12"),
            ("4", "12"),
            ("5", "12"),
            ("6", "12"),
            ("7", "12"),
            ("8", "12"),
            ("9", "12"),
            ("</s>", "12"),
            ("<s>", "12"),
        }:
            return 31
        elif key in {
            ("1", "16"),
            ("1", "24"),
            ("11", "16"),
            ("15", "16"),
            ("16", "14"),
            ("16", "15"),
            ("16", "19"),
            ("16", "20"),
            ("16", "23"),
            ("16", "24"),
            ("16", "25"),
            ("16", "27"),
            ("16", "28"),
            ("16", "3"),
            ("16", "5"),
            ("16", "6"),
            ("16", "7"),
            ("16", "8"),
            ("16", "9"),
            ("17", "16"),
            ("17", "24"),
            ("19", "16"),
            ("22", "16"),
            ("23", "16"),
            ("23", "24"),
            ("24", "16"),
            ("25", "16"),
            ("28", "16"),
            ("28", "24"),
            ("3", "16"),
            ("3", "24"),
            ("4", "16"),
            ("5", "16"),
            ("5", "24"),
            ("6", "16"),
            ("6", "24"),
            ("</s>", "16"),
            ("</s>", "24"),
            ("<s>", "16"),
        }:
            return 16
        elif key in {
            ("0", "0"),
            ("1", "0"),
            ("10", "0"),
            ("11", "0"),
            ("14", "0"),
            ("14", "4"),
            ("16", "0"),
            ("18", "0"),
            ("19", "0"),
            ("20", "0"),
            ("20", "4"),
            ("22", "0"),
            ("23", "0"),
            ("24", "0"),
            ("25", "0"),
            ("25", "4"),
            ("26", "0"),
            ("28", "0"),
            ("4", "0"),
            ("5", "0"),
            ("6", "0"),
            ("7", "0"),
            ("8", "0"),
            ("9", "0"),
            ("9", "14"),
            ("9", "22"),
            ("9", "5"),
            ("9", "7"),
            ("9", "9"),
            ("</s>", "0"),
            ("<s>", "0"),
        }:
            return 4
        elif key in {
            ("14", "16"),
            ("14", "24"),
            ("16", "22"),
            ("16", "26"),
            ("16", "4"),
            ("16", "<s>"),
            ("19", "24"),
            ("20", "24"),
            ("25", "24"),
            ("26", "16"),
            ("26", "24"),
            ("4", "24"),
            ("4", "4"),
            ("9", "24"),
            ("9", "4"),
            ("<s>", "24"),
        }:
            return 11
        elif key in {
            ("0", "16"),
            ("16", "1"),
            ("16", "10"),
            ("16", "16"),
            ("16", "17"),
            ("16", "18"),
            ("16", "2"),
            ("16", "</s>"),
            ("18", "16"),
            ("2", "16"),
            ("20", "16"),
            ("21", "16"),
            ("7", "16"),
            ("8", "16"),
            ("9", "16"),
        }:
            return 26
        elif key in {("12", "16")}:
            return 13
        return 2

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_1_3_output):
        key = (num_attn_2_0_output, num_attn_1_3_output)
        return 22

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_2_3_output):
        key = (num_attn_1_1_output, num_attn_2_3_output)
        return 14

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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


print(run(["<s>", "19", "22", "6", "26", "26", "21", "</s>"]))

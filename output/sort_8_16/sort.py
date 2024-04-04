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
        "output/sort_8_16/sort_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_token, k_token):
        if q_token in {"17", "1", "10", "15", "0", "13", "11"}:
            return k_token == "12"
        elif q_token in {"12", "14"}:
            return k_token == "13"
        elif q_token in {"16"}:
            return k_token == "15"
        elif q_token in {"27", "24", "18"}:
            return k_token == "4"
        elif q_token in {"19", "21", "22"}:
            return k_token == "20"
        elif q_token in {"2", "</s>", "23"}:
            return k_token == "21"
        elif q_token in {"26", "20"}:
            return k_token == "16"
        elif q_token in {"25"}:
            return k_token == "3"
        elif q_token in {"28", "8"}:
            return k_token == "27"
        elif q_token in {"9", "3"}:
            return k_token == "5"
        elif q_token in {"5", "4"}:
            return k_token == "28"
        elif q_token in {"6"}:
            return k_token == "7"
        elif q_token in {"7"}:
            return k_token == "6"
        elif q_token in {"<s>"}:
            return k_token == "24"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 8}:
            return token == "3"
        elif position in {1}:
            return token == "0"
        elif position in {2, 30}:
            return token == "12"
        elif position in {3}:
            return token == "19"
        elif position in {4, 5}:
            return token == "13"
        elif position in {6}:
            return token == "15"
        elif position in {9, 7}:
            return token == "23"
        elif position in {10}:
            return token == "6"
        elif position in {11, 13, 14}:
            return token == "4"
        elif position in {24, 26, 12, 29}:
            return token == "7"
        elif position in {17, 15}:
            return token == "25"
        elif position in {16, 18, 19, 21, 22, 23, 25}:
            return token == "26"
        elif position in {20}:
            return token == "21"
        elif position in {27}:
            return token == "28"
        elif position in {28}:
            return token == "9"
        elif position in {31}:
            return token == "18"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 18, 26}:
            return token == "9"
        elif position in {1}:
            return token == "11"
        elif position in {2, 3}:
            return token == "14"
        elif position in {4}:
            return token == "2"
        elif position in {5}:
            return token == "0"
        elif position in {6, 15}:
            return token == "26"
        elif position in {7}:
            return token == "22"
        elif position in {8, 9}:
            return token == "7"
        elif position in {17, 10, 27, 23}:
            return token == "28"
        elif position in {11, 12, 13, 14, 21, 24}:
            return token == "6"
        elif position in {16}:
            return token == "27"
        elif position in {19}:
            return token == "12"
        elif position in {20}:
            return token == "8"
        elif position in {22}:
            return token == "5"
        elif position in {25}:
            return token == "3"
        elif position in {28, 29}:
            return token == "4"
        elif position in {30}:
            return token == "18"
        elif position in {31}:
            return token == "15"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 30, 31}:
            return token == "12"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 4, 15}:
            return token == "0"
        elif position in {5}:
            return token == "14"
        elif position in {6}:
            return token == "22"
        elif position in {17, 11, 25, 7}:
            return token == "7"
        elif position in {8, 21}:
            return token == "4"
        elif position in {9, 10, 12, 13, 14, 20, 27, 29}:
            return token == "9"
        elif position in {16}:
            return token == "23"
        elif position in {18, 26}:
            return token == "8"
        elif position in {19}:
            return token == "2"
        elif position in {22, 23}:
            return token == "5"
        elif position in {24, 28}:
            return token == "3"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 2, 4, 15, 16}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {3, 18, 20, 25, 27, 30, 31}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {6}:
            return k_position == 24
        elif q_position in {7}:
            return k_position == 25
        elif q_position in {8}:
            return k_position == 19
        elif q_position in {9, 12}:
            return k_position == 10
        elif q_position in {10, 11}:
            return k_position == 5
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 3
        elif q_position in {17, 19, 21, 23, 24, 26}:
            return k_position == 15
        elif q_position in {22}:
            return k_position == 16
        elif q_position in {28}:
            return k_position == 31
        elif q_position in {29}:
            return k_position == 23

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 10, 21, 30}:
            return k_position == 13
        elif q_position in {1, 2, 15}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 0
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {8, 26, 7}:
            return k_position == 11
        elif q_position in {9, 16, 17, 18, 19, 20, 22, 23, 28, 31}:
            return k_position == 12
        elif q_position in {25, 11}:
            return k_position == 14
        elif q_position in {24, 27, 12, 29}:
            return k_position == 15
        elif q_position in {13, 14}:
            return k_position == 29

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 31, 22, 30}:
            return k_position == 12
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {18, 4, 5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {7, 8, 17, 23, 27}:
            return k_position == 13
        elif q_position in {9, 10, 16, 19, 20, 24, 25, 26, 28, 29}:
            return k_position == 14
        elif q_position in {11, 21}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 27
        elif q_position in {14}:
            return k_position == 25
        elif q_position in {15}:
            return k_position == 8

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 3, 4, 6}:
            return k_position == 10
        elif q_position in {7, 8, 14, 15, 16, 17, 22, 25, 27, 31}:
            return k_position == 12
        elif q_position in {9, 18, 19, 20, 21, 23, 24, 26, 28, 30}:
            return k_position == 13
        elif q_position in {10, 29}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 25
        elif q_position in {13}:
            return k_position == 8

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {0, 6, 7, 9, 18, 22, 25, 30}:
            return 7
        elif key in {4, 5, 24, 26, 27, 31}:
            return 6
        elif key in {1, 2, 3, 15, 20}:
            return 2
        return 17

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "1"),
            (0, "10"),
            (0, "11"),
            (0, "12"),
            (0, "13"),
            (0, "14"),
            (0, "15"),
            (0, "16"),
            (0, "17"),
            (0, "18"),
            (0, "19"),
            (0, "2"),
            (0, "20"),
            (0, "21"),
            (0, "22"),
            (0, "23"),
            (0, "24"),
            (0, "25"),
            (0, "26"),
            (0, "27"),
            (0, "28"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "6"),
            (0, "7"),
            (0, "8"),
            (0, "9"),
            (0, "</s>"),
            (0, "<s>"),
            (2, "1"),
            (2, "10"),
            (2, "11"),
            (2, "12"),
            (2, "13"),
            (2, "14"),
            (2, "15"),
            (2, "16"),
            (2, "17"),
            (2, "18"),
            (2, "19"),
            (2, "2"),
            (2, "20"),
            (2, "21"),
            (2, "22"),
            (2, "23"),
            (2, "24"),
            (2, "25"),
            (2, "26"),
            (2, "27"),
            (2, "28"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "6"),
            (2, "7"),
            (2, "8"),
            (2, "9"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "1"),
            (3, "10"),
            (3, "11"),
            (3, "12"),
            (3, "13"),
            (3, "14"),
            (3, "15"),
            (3, "16"),
            (3, "17"),
            (3, "18"),
            (3, "19"),
            (3, "2"),
            (3, "20"),
            (3, "21"),
            (3, "22"),
            (3, "23"),
            (3, "24"),
            (3, "25"),
            (3, "26"),
            (3, "27"),
            (3, "28"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "6"),
            (3, "7"),
            (3, "8"),
            (3, "9"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "1"),
            (4, "10"),
            (4, "11"),
            (4, "12"),
            (4, "13"),
            (4, "14"),
            (4, "15"),
            (4, "16"),
            (4, "17"),
            (4, "18"),
            (4, "19"),
            (4, "2"),
            (4, "20"),
            (4, "21"),
            (4, "22"),
            (4, "23"),
            (4, "24"),
            (4, "25"),
            (4, "26"),
            (4, "27"),
            (4, "28"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "6"),
            (4, "7"),
            (4, "8"),
            (4, "9"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "1"),
            (6, "1"),
            (7, "1"),
            (7, "12"),
            (8, "1"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
            (12, "1"),
            (13, "1"),
            (14, "1"),
            (15, "1"),
            (15, "10"),
            (15, "11"),
            (16, "1"),
            (16, "10"),
            (16, "11"),
            (16, "12"),
            (16, "15"),
            (16, "16"),
            (16, "18"),
            (16, "20"),
            (16, "23"),
            (16, "25"),
            (16, "26"),
            (16, "27"),
            (16, "28"),
            (16, "7"),
            (16, "9"),
            (17, "1"),
            (17, "10"),
            (17, "11"),
            (17, "12"),
            (17, "13"),
            (17, "14"),
            (17, "15"),
            (17, "16"),
            (17, "17"),
            (17, "18"),
            (17, "19"),
            (17, "2"),
            (17, "20"),
            (17, "21"),
            (17, "23"),
            (17, "24"),
            (17, "25"),
            (17, "26"),
            (17, "27"),
            (17, "28"),
            (17, "3"),
            (17, "4"),
            (17, "5"),
            (17, "6"),
            (17, "7"),
            (17, "8"),
            (17, "9"),
            (18, "1"),
            (18, "10"),
            (18, "12"),
            (18, "25"),
            (18, "27"),
            (18, "7"),
            (18, "9"),
            (19, "1"),
            (19, "10"),
            (19, "11"),
            (19, "12"),
            (19, "15"),
            (19, "16"),
            (19, "18"),
            (19, "19"),
            (19, "20"),
            (19, "23"),
            (19, "25"),
            (19, "26"),
            (19, "27"),
            (19, "28"),
            (19, "7"),
            (19, "9"),
            (20, "1"),
            (20, "10"),
            (20, "12"),
            (20, "27"),
            (20, "7"),
            (21, "1"),
            (21, "10"),
            (21, "11"),
            (21, "12"),
            (21, "15"),
            (21, "16"),
            (21, "18"),
            (21, "19"),
            (21, "20"),
            (21, "23"),
            (21, "25"),
            (21, "26"),
            (21, "27"),
            (21, "28"),
            (21, "5"),
            (21, "7"),
            (21, "9"),
            (22, "1"),
            (22, "10"),
            (22, "11"),
            (22, "12"),
            (22, "15"),
            (22, "16"),
            (22, "18"),
            (22, "19"),
            (22, "20"),
            (22, "23"),
            (22, "25"),
            (22, "26"),
            (22, "27"),
            (22, "28"),
            (22, "7"),
            (22, "9"),
            (23, "1"),
            (23, "10"),
            (23, "11"),
            (23, "12"),
            (23, "15"),
            (23, "16"),
            (23, "18"),
            (23, "19"),
            (23, "20"),
            (23, "23"),
            (23, "25"),
            (23, "26"),
            (23, "27"),
            (23, "28"),
            (23, "5"),
            (23, "6"),
            (23, "7"),
            (23, "8"),
            (23, "9"),
            (24, "1"),
            (24, "10"),
            (24, "11"),
            (24, "12"),
            (24, "13"),
            (24, "14"),
            (24, "15"),
            (24, "16"),
            (24, "17"),
            (24, "18"),
            (24, "19"),
            (24, "2"),
            (24, "20"),
            (24, "21"),
            (24, "23"),
            (24, "24"),
            (24, "25"),
            (24, "26"),
            (24, "27"),
            (24, "28"),
            (24, "3"),
            (24, "4"),
            (24, "5"),
            (24, "6"),
            (24, "7"),
            (24, "8"),
            (24, "9"),
            (25, "1"),
            (25, "10"),
            (25, "11"),
            (25, "12"),
            (25, "15"),
            (25, "16"),
            (25, "18"),
            (25, "19"),
            (25, "20"),
            (25, "23"),
            (25, "25"),
            (25, "26"),
            (25, "27"),
            (25, "5"),
            (25, "7"),
            (25, "9"),
            (26, "1"),
            (26, "10"),
            (27, "1"),
            (27, "10"),
            (27, "11"),
            (27, "12"),
            (27, "15"),
            (27, "16"),
            (27, "18"),
            (27, "19"),
            (27, "2"),
            (27, "20"),
            (27, "23"),
            (27, "24"),
            (27, "25"),
            (27, "26"),
            (27, "27"),
            (27, "28"),
            (27, "5"),
            (27, "7"),
            (27, "9"),
            (28, "1"),
            (28, "10"),
            (28, "11"),
            (28, "12"),
            (28, "15"),
            (28, "16"),
            (28, "18"),
            (28, "19"),
            (28, "20"),
            (28, "23"),
            (28, "25"),
            (28, "26"),
            (28, "27"),
            (28, "28"),
            (28, "5"),
            (28, "6"),
            (28, "7"),
            (28, "8"),
            (28, "9"),
            (29, "1"),
            (29, "10"),
            (29, "11"),
            (29, "12"),
            (29, "15"),
            (29, "16"),
            (29, "18"),
            (29, "19"),
            (29, "20"),
            (29, "23"),
            (29, "25"),
            (29, "26"),
            (29, "27"),
            (29, "5"),
            (29, "7"),
            (29, "9"),
            (30, "1"),
            (30, "10"),
            (30, "12"),
            (30, "15"),
            (30, "16"),
            (30, "20"),
            (30, "25"),
            (30, "26"),
            (30, "27"),
            (30, "7"),
            (30, "9"),
            (31, "1"),
            (31, "10"),
            (31, "11"),
            (31, "12"),
            (31, "14"),
            (31, "15"),
            (31, "16"),
            (31, "17"),
            (31, "18"),
            (31, "19"),
            (31, "2"),
            (31, "20"),
            (31, "21"),
            (31, "23"),
            (31, "24"),
            (31, "25"),
            (31, "26"),
            (31, "27"),
            (31, "28"),
            (31, "4"),
            (31, "5"),
            (31, "6"),
            (31, "7"),
            (31, "8"),
            (31, "9"),
        }:
            return 11
        elif key in {
            (1, "0"),
            (1, "12"),
            (1, "13"),
            (1, "14"),
            (1, "15"),
            (1, "16"),
            (1, "17"),
            (1, "18"),
            (1, "19"),
            (1, "2"),
            (1, "20"),
            (1, "21"),
            (1, "22"),
            (1, "23"),
            (1, "24"),
            (1, "25"),
            (1, "26"),
            (1, "27"),
            (1, "28"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
            (13, "0"),
            (14, "0"),
            (15, "0"),
            (15, "12"),
            (15, "13"),
            (15, "15"),
            (15, "16"),
            (15, "17"),
            (15, "18"),
            (15, "19"),
            (15, "2"),
            (15, "20"),
            (15, "21"),
            (15, "23"),
            (15, "24"),
            (15, "25"),
            (15, "26"),
            (15, "27"),
            (15, "28"),
            (15, "3"),
            (15, "5"),
            (15, "6"),
            (15, "7"),
            (15, "8"),
            (15, "9"),
            (18, "0"),
            (18, "20"),
            (19, "0"),
            (20, "0"),
            (26, "0"),
            (27, "13"),
            (28, "0"),
            (29, "0"),
            (31, "0"),
            (31, "13"),
        }:
            return 10
        elif key in {
            (0, "0"),
            (1, "1"),
            (1, "10"),
            (1, "11"),
            (2, "0"),
            (3, "0"),
            (4, "0"),
            (5, "0"),
            (6, "0"),
            (7, "0"),
            (8, "0"),
            (9, "0"),
            (10, "0"),
            (11, "0"),
            (12, "0"),
            (16, "0"),
            (17, "0"),
            (21, "0"),
            (22, "0"),
            (23, "0"),
            (24, "0"),
            (25, "0"),
            (27, "0"),
            (30, "0"),
        }:
            return 24
        elif key in {(5, "15"), (5, "5"), (5, "7"), (5, "9"), (7, "7")}:
            return 7
        return 15

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 23

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 3

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0}:
            return token == "16"
        elif position in {1, 18, 25}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3, 30}:
            return token == "20"
        elif position in {4, 20, 15}:
            return token == "12"
        elif position in {5}:
            return token == "18"
        elif position in {24, 6}:
            return token == "23"
        elif position in {7}:
            return token == "2"
        elif position in {8, 9, 10, 11}:
            return token == "5"
        elif position in {12, 13, 23}:
            return token == "15"
        elif position in {14}:
            return token == "27"
        elif position in {16}:
            return token == "7"
        elif position in {17}:
            return token == "3"
        elif position in {19}:
            return token == "11"
        elif position in {29, 21}:
            return token == "22"
        elif position in {28, 22}:
            return token == "6"
        elif position in {26}:
            return token == "26"
        elif position in {27}:
            return token == "28"
        elif position in {31}:
            return token == "21"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0}:
            return token == "18"
        elif position in {1}:
            return token == "12"
        elif position in {2, 3}:
            return token == "15"
        elif position in {4}:
            return token == "14"
        elif position in {27, 5}:
            return token == "2"
        elif position in {18, 6}:
            return token == "25"
        elif position in {8, 11, 28, 7}:
            return token == "9"
        elif position in {9}:
            return token == "4"
        elif position in {10, 13, 14, 17, 29}:
            return token == "7"
        elif position in {26, 12}:
            return token == "5"
        elif position in {21, 15}:
            return token == "20"
        elif position in {16}:
            return token == "26"
        elif position in {19}:
            return token == "22"
        elif position in {20}:
            return token == "16"
        elif position in {22}:
            return token == "8"
        elif position in {23}:
            return token == "27"
        elif position in {24}:
            return token == "3"
        elif position in {25}:
            return token == "6"
        elif position in {30}:
            return token == "13"
        elif position in {31}:
            return token == "24"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 6}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {1}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {3, 4}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {5}:
            return mlp_0_1_output == 26
        elif mlp_0_0_output in {8, 17, 29, 7}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 22
        elif mlp_0_0_output in {10}:
            return mlp_0_1_output == 27
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 23
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 25
        elif mlp_0_0_output in {13}:
            return mlp_0_1_output == 29
        elif mlp_0_0_output in {14}:
            return mlp_0_1_output == 28
        elif mlp_0_0_output in {15, 18, 21, 26, 31}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {16, 19}:
            return mlp_0_1_output == 3
        elif mlp_0_0_output in {20}:
            return mlp_0_1_output == 14
        elif mlp_0_0_output in {22}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {23}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {24, 27, 28, 30}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {25}:
            return mlp_0_1_output == 9

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2, 20}:
            return k_position == 3
        elif q_position in {11, 3, 15}:
            return k_position == 5
        elif q_position in {4, 6, 7}:
            return k_position == 8
        elif q_position in {5, 10, 16, 21, 24, 28, 30, 31}:
            return k_position == 6
        elif q_position in {8, 18, 19, 14}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 27
        elif q_position in {13}:
            return k_position == 28
        elif q_position in {17}:
            return k_position == 26
        elif q_position in {26, 22}:
            return k_position == 12
        elif q_position in {23}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 9
        elif q_position in {27}:
            return k_position == 24
        elif q_position in {29}:
            return k_position == 11

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, mlp_0_0_output):
        if position in {0, 31}:
            return mlp_0_0_output == 6
        elif position in {1, 4}:
            return mlp_0_0_output == 2
        elif position in {2, 5, 6, 10, 20, 30}:
            return mlp_0_0_output == 17
        elif position in {3, 7, 8, 16, 18}:
            return mlp_0_0_output == 7
        elif position in {9, 19}:
            return mlp_0_0_output == 15
        elif position in {11, 12, 14, 17, 23, 26, 29}:
            return mlp_0_0_output == 12
        elif position in {13}:
            return mlp_0_0_output == 1
        elif position in {15}:
            return mlp_0_0_output == 11
        elif position in {27, 21, 22}:
            return mlp_0_0_output == 13
        elif position in {24, 25, 28}:
            return mlp_0_0_output == 3

    num_attn_1_0_pattern = select(mlp_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_position, k_position):
        if q_position in {0, 10, 18, 21, 23, 29, 30, 31}:
            return k_position == 12
        elif q_position in {1, 2, 3, 4, 5, 6}:
            return k_position == 1
        elif q_position in {8, 7}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {17, 26, 11}:
            return k_position == 13
        elif q_position in {25, 27, 12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15, 16, 19, 20, 22, 24, 28}:
            return k_position == 17

    num_attn_1_1_pattern = select(positions, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0}:
            return token == "12"
        elif position in {24, 1, 2}:
            return token == "0"
        elif position in {3, 4, 5, 6, 7, 8, 12, 16, 19, 22, 23, 26, 27, 28, 30, 31}:
            return token == "17"
        elif position in {9}:
            return token == "14"
        elif position in {10, 29}:
            return token == "19"
        elif position in {11}:
            return token == "2"
        elif position in {13}:
            return token == "25"
        elif position in {14}:
            return token == "23"
        elif position in {15}:
            return token == "13"
        elif position in {17, 20}:
            return token == "20"
        elif position in {25, 18, 21}:
            return token == "18"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_position, k_position):
        if q_position in {0, 18, 5, 6}:
            return k_position == 7
        elif q_position in {24, 1, 2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 15}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {17, 11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 22
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {16, 19, 21, 22, 23, 25, 26, 28, 30, 31}:
            return k_position == 11
        elif q_position in {27, 20, 29}:
            return k_position == 12

    num_attn_1_3_pattern = select(positions, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 14),
            ("0", 22),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 14),
            ("10", 8),
            ("10", 9),
            ("10", 10),
            ("10", 11),
            ("10", 12),
            ("10", 13),
            ("10", 14),
            ("11", 8),
            ("11", 9),
            ("11", 10),
            ("11", 11),
            ("11", 12),
            ("11", 13),
            ("11", 14),
            ("11", 22),
            ("11", 29),
            ("12", 8),
            ("12", 9),
            ("12", 10),
            ("12", 11),
            ("12", 12),
            ("12", 13),
            ("12", 14),
            ("12", 17),
            ("12", 29),
            ("13", 8),
            ("13", 9),
            ("13", 10),
            ("13", 11),
            ("13", 12),
            ("13", 13),
            ("13", 14),
            ("13", 22),
            ("13", 29),
            ("14", 8),
            ("14", 9),
            ("14", 10),
            ("14", 11),
            ("14", 12),
            ("14", 13),
            ("14", 14),
            ("15", 8),
            ("15", 9),
            ("15", 10),
            ("15", 11),
            ("15", 12),
            ("15", 13),
            ("15", 14),
            ("16", 8),
            ("16", 9),
            ("16", 10),
            ("16", 11),
            ("16", 12),
            ("16", 13),
            ("16", 14),
            ("16", 17),
            ("16", 22),
            ("16", 29),
            ("17", 0),
            ("17", 8),
            ("17", 9),
            ("17", 10),
            ("17", 11),
            ("17", 12),
            ("17", 13),
            ("17", 14),
            ("17", 22),
            ("17", 25),
            ("17", 28),
            ("17", 29),
            ("18", 8),
            ("18", 9),
            ("18", 10),
            ("18", 11),
            ("18", 12),
            ("18", 13),
            ("18", 14),
            ("19", 0),
            ("19", 8),
            ("19", 9),
            ("19", 10),
            ("19", 11),
            ("19", 12),
            ("19", 13),
            ("19", 14),
            ("19", 20),
            ("19", 21),
            ("19", 22),
            ("19", 23),
            ("19", 25),
            ("19", 28),
            ("19", 29),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("2", 14),
            ("2", 22),
            ("2", 29),
            ("20", 0),
            ("20", 8),
            ("20", 9),
            ("20", 10),
            ("20", 11),
            ("20", 12),
            ("20", 13),
            ("20", 14),
            ("20", 21),
            ("20", 22),
            ("20", 23),
            ("20", 25),
            ("20", 28),
            ("20", 29),
            ("21", 8),
            ("21", 9),
            ("21", 10),
            ("21", 11),
            ("21", 12),
            ("21", 13),
            ("21", 14),
            ("21", 22),
            ("21", 28),
            ("21", 29),
            ("22", 0),
            ("22", 8),
            ("22", 9),
            ("22", 10),
            ("22", 11),
            ("22", 12),
            ("22", 13),
            ("22", 14),
            ("22", 22),
            ("22", 23),
            ("22", 25),
            ("22", 28),
            ("22", 29),
            ("23", 0),
            ("23", 8),
            ("23", 9),
            ("23", 10),
            ("23", 11),
            ("23", 12),
            ("23", 13),
            ("23", 14),
            ("23", 20),
            ("23", 22),
            ("23", 23),
            ("23", 25),
            ("23", 29),
            ("24", 8),
            ("24", 9),
            ("24", 10),
            ("24", 11),
            ("24", 12),
            ("24", 13),
            ("24", 14),
            ("24", 17),
            ("24", 22),
            ("24", 28),
            ("24", 29),
            ("25", 0),
            ("25", 8),
            ("25", 9),
            ("25", 10),
            ("25", 11),
            ("25", 12),
            ("25", 13),
            ("25", 14),
            ("25", 20),
            ("25", 22),
            ("25", 25),
            ("25", 29),
            ("26", 8),
            ("26", 9),
            ("26", 10),
            ("26", 11),
            ("26", 12),
            ("26", 13),
            ("26", 14),
            ("26", 22),
            ("26", 29),
            ("27", 8),
            ("27", 9),
            ("27", 10),
            ("27", 11),
            ("27", 12),
            ("27", 13),
            ("27", 14),
            ("28", 0),
            ("28", 8),
            ("28", 9),
            ("28", 10),
            ("28", 11),
            ("28", 12),
            ("28", 13),
            ("28", 14),
            ("28", 15),
            ("28", 22),
            ("28", 25),
            ("28", 28),
            ("28", 29),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("4", 0),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 22),
            ("4", 29),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("5", 14),
            ("5", 22),
            ("6", 8),
            ("6", 9),
            ("6", 10),
            ("6", 11),
            ("6", 12),
            ("6", 13),
            ("6", 14),
            ("7", 0),
            ("7", 8),
            ("7", 9),
            ("7", 10),
            ("7", 11),
            ("7", 12),
            ("7", 13),
            ("7", 14),
            ("7", 22),
            ("7", 29),
            ("8", 8),
            ("8", 9),
            ("8", 10),
            ("8", 11),
            ("8", 12),
            ("8", 13),
            ("8", 14),
            ("9", 0),
            ("9", 8),
            ("9", 9),
            ("9", 10),
            ("9", 11),
            ("9", 12),
            ("9", 13),
            ("9", 14),
            ("9", 15),
            ("9", 16),
            ("9", 20),
            ("9", 21),
            ("9", 22),
            ("9", 23),
            ("9", 25),
            ("9", 28),
            ("9", 29),
            ("9", 30),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 13),
            ("</s>", 14),
            ("</s>", 22),
            ("</s>", 29),
            ("<s>", 0),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 22),
            ("<s>", 29),
        }:
            return 11
        elif key in {
            ("0", 0),
            ("0", 2),
            ("0", 16),
            ("0", 20),
            ("0", 21),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 29),
            ("1", 0),
            ("1", 2),
            ("1", 15),
            ("1", 16),
            ("1", 18),
            ("1", 20),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 25),
            ("1", 26),
            ("1", 28),
            ("1", 29),
            ("1", 30),
            ("1", 31),
            ("10", 2),
            ("10", 16),
            ("12", 2),
            ("12", 16),
            ("12", 20),
            ("12", 21),
            ("12", 24),
            ("12", 25),
            ("12", 27),
            ("13", 2),
            ("15", 0),
            ("15", 2),
            ("15", 16),
            ("15", 20),
            ("15", 21),
            ("15", 24),
            ("16", 2),
            ("17", 2),
            ("17", 16),
            ("17", 21),
            ("17", 24),
            ("18", 2),
            ("18", 16),
            ("18", 21),
            ("18", 24),
            ("19", 2),
            ("19", 16),
            ("2", 0),
            ("2", 2),
            ("2", 16),
            ("2", 20),
            ("2", 21),
            ("2", 23),
            ("2", 24),
            ("2", 25),
            ("20", 2),
            ("20", 16),
            ("20", 20),
            ("20", 24),
            ("21", 0),
            ("21", 2),
            ("21", 16),
            ("21", 20),
            ("21", 21),
            ("21", 24),
            ("21", 25),
            ("22", 2),
            ("22", 16),
            ("22", 20),
            ("22", 21),
            ("22", 24),
            ("23", 2),
            ("24", 0),
            ("24", 2),
            ("24", 15),
            ("24", 16),
            ("24", 18),
            ("24", 20),
            ("24", 21),
            ("24", 23),
            ("24", 24),
            ("24", 25),
            ("24", 26),
            ("24", 27),
            ("24", 30),
            ("24", 31),
            ("25", 2),
            ("26", 0),
            ("26", 2),
            ("26", 16),
            ("26", 20),
            ("26", 21),
            ("26", 24),
            ("26", 25),
            ("27", 2),
            ("28", 2),
            ("28", 16),
            ("28", 20),
            ("28", 21),
            ("28", 24),
            ("3", 2),
            ("4", 2),
            ("4", 16),
            ("4", 20),
            ("5", 2),
            ("5", 16),
            ("5", 20),
            ("5", 21),
            ("5", 24),
            ("6", 2),
            ("6", 16),
            ("6", 20),
            ("7", 2),
            ("7", 16),
            ("7", 20),
            ("7", 21),
            ("7", 24),
            ("8", 2),
            ("8", 16),
            ("8", 20),
            ("8", 21),
            ("8", 24),
            ("9", 2),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 16),
            ("</s>", 20),
            ("</s>", 21),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 25),
            ("<s>", 2),
            ("<s>", 16),
            ("<s>", 20),
        }:
            return 28
        elif key in {
            ("0", 1),
            ("1", 1),
            ("10", 1),
            ("11", 0),
            ("11", 1),
            ("11", 2),
            ("11", 15),
            ("11", 16),
            ("11", 20),
            ("11", 21),
            ("11", 23),
            ("11", 24),
            ("11", 25),
            ("11", 30),
            ("12", 0),
            ("12", 1),
            ("12", 15),
            ("12", 18),
            ("12", 19),
            ("12", 22),
            ("12", 23),
            ("12", 26),
            ("12", 28),
            ("12", 30),
            ("12", 31),
            ("13", 0),
            ("13", 1),
            ("13", 15),
            ("13", 16),
            ("13", 18),
            ("13", 20),
            ("13", 21),
            ("13", 23),
            ("13", 24),
            ("13", 25),
            ("13", 30),
            ("13", 31),
            ("14", 0),
            ("14", 1),
            ("14", 2),
            ("14", 16),
            ("14", 20),
            ("14", 22),
            ("14", 23),
            ("14", 25),
            ("14", 28),
            ("14", 29),
            ("15", 1),
            ("15", 23),
            ("16", 0),
            ("16", 1),
            ("16", 15),
            ("16", 16),
            ("16", 18),
            ("16", 19),
            ("16", 20),
            ("16", 21),
            ("16", 23),
            ("16", 24),
            ("16", 25),
            ("16", 26),
            ("16", 27),
            ("16", 28),
            ("16", 30),
            ("16", 31),
            ("17", 1),
            ("17", 20),
            ("17", 23),
            ("18", 0),
            ("18", 1),
            ("18", 20),
            ("18", 22),
            ("18", 23),
            ("18", 25),
            ("18", 26),
            ("18", 27),
            ("18", 28),
            ("18", 29),
            ("18", 31),
            ("19", 1),
            ("2", 1),
            ("20", 1),
            ("21", 1),
            ("21", 23),
            ("22", 1),
            ("23", 1),
            ("24", 1),
            ("25", 1),
            ("26", 1),
            ("27", 1),
            ("28", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("6", 1),
            ("7", 1),
            ("8", 1),
            ("9", 1),
            ("</s>", 1),
            ("<s>", 1),
        }:
            return 2
        elif key in {
            ("0", 3),
            ("1", 3),
            ("10", 0),
            ("10", 3),
            ("10", 18),
            ("10", 20),
            ("10", 23),
            ("10", 30),
            ("10", 31),
            ("11", 3),
            ("11", 4),
            ("11", 18),
            ("11", 19),
            ("11", 26),
            ("11", 27),
            ("11", 31),
            ("12", 3),
            ("12", 4),
            ("13", 3),
            ("13", 4),
            ("13", 19),
            ("14", 3),
            ("14", 4),
            ("14", 5),
            ("14", 15),
            ("14", 18),
            ("14", 19),
            ("14", 21),
            ("14", 24),
            ("14", 26),
            ("14", 27),
            ("14", 30),
            ("14", 31),
            ("15", 3),
            ("16", 3),
            ("16", 4),
            ("16", 5),
            ("17", 3),
            ("17", 4),
            ("17", 15),
            ("17", 18),
            ("17", 19),
            ("17", 26),
            ("17", 30),
            ("17", 31),
            ("18", 3),
            ("18", 4),
            ("18", 5),
            ("18", 15),
            ("18", 18),
            ("18", 19),
            ("18", 30),
            ("19", 3),
            ("2", 3),
            ("20", 3),
            ("21", 3),
            ("21", 4),
            ("21", 15),
            ("21", 18),
            ("21", 19),
            ("21", 30),
            ("21", 31),
            ("22", 3),
            ("24", 3),
            ("25", 3),
            ("26", 3),
            ("27", 3),
            ("28", 3),
            ("3", 3),
            ("4", 3),
            ("5", 3),
            ("6", 3),
            ("7", 3),
            ("9", 3),
            ("</s>", 3),
            ("<s>", 3),
        }:
            return 23
        elif key in {
            ("10", 22),
            ("11", 5),
            ("11", 6),
            ("11", 7),
            ("11", 17),
            ("11", 28),
            ("12", 5),
            ("12", 6),
            ("12", 7),
            ("13", 5),
            ("13", 6),
            ("13", 7),
            ("13", 17),
            ("13", 26),
            ("13", 27),
            ("13", 28),
            ("15", 22),
            ("17", 27),
            ("21", 26),
        }:
            return 5
        elif key in {("26", 23), ("28", 23), ("5", 0), ("5", 23), ("5", 25), ("5", 29)}:
            return 30
        return 16

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, attn_1_3_output):
        key = (attn_1_0_output, attn_1_3_output)
        if key in {
            ("0", "2"),
            ("0", "23"),
            ("0", "8"),
            ("1", "2"),
            ("1", "23"),
            ("1", "8"),
            ("10", "2"),
            ("10", "23"),
            ("10", "8"),
            ("11", "2"),
            ("11", "23"),
            ("11", "8"),
            ("12", "2"),
            ("12", "23"),
            ("12", "8"),
            ("13", "2"),
            ("13", "23"),
            ("13", "8"),
            ("14", "2"),
            ("14", "23"),
            ("14", "8"),
            ("15", "8"),
            ("16", "2"),
            ("16", "23"),
            ("16", "8"),
            ("17", "2"),
            ("17", "23"),
            ("17", "8"),
            ("18", "2"),
            ("18", "23"),
            ("18", "8"),
            ("19", "2"),
            ("19", "23"),
            ("19", "8"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "12"),
            ("2", "13"),
            ("2", "14"),
            ("2", "16"),
            ("2", "17"),
            ("2", "18"),
            ("2", "19"),
            ("2", "2"),
            ("2", "20"),
            ("2", "21"),
            ("2", "22"),
            ("2", "23"),
            ("2", "24"),
            ("2", "25"),
            ("2", "26"),
            ("2", "27"),
            ("2", "28"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "7"),
            ("2", "8"),
            ("2", "9"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("20", "2"),
            ("20", "23"),
            ("20", "8"),
            ("21", "2"),
            ("21", "23"),
            ("21", "8"),
            ("22", "2"),
            ("22", "23"),
            ("22", "8"),
            ("23", "2"),
            ("23", "23"),
            ("23", "8"),
            ("24", "2"),
            ("24", "23"),
            ("24", "8"),
            ("25", "23"),
            ("25", "8"),
            ("26", "2"),
            ("26", "23"),
            ("26", "8"),
            ("27", "1"),
            ("27", "2"),
            ("27", "23"),
            ("27", "8"),
            ("28", "2"),
            ("28", "23"),
            ("28", "8"),
            ("3", "19"),
            ("3", "2"),
            ("3", "23"),
            ("3", "8"),
            ("3", "9"),
            ("4", "23"),
            ("4", "8"),
            ("5", "2"),
            ("5", "23"),
            ("5", "8"),
            ("6", "2"),
            ("6", "23"),
            ("6", "8"),
            ("7", "2"),
            ("7", "8"),
            ("8", "2"),
            ("8", "23"),
            ("8", "8"),
            ("9", "19"),
            ("9", "2"),
            ("9", "23"),
            ("9", "8"),
            ("9", "9"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "23"),
            ("</s>", "8"),
            ("<s>", "2"),
            ("<s>", "23"),
            ("<s>", "8"),
        }:
            return 5
        elif key in {
            ("0", "15"),
            ("1", "15"),
            ("10", "15"),
            ("11", "15"),
            ("12", "15"),
            ("13", "15"),
            ("14", "15"),
            ("15", "0"),
            ("15", "1"),
            ("15", "10"),
            ("15", "11"),
            ("15", "12"),
            ("15", "13"),
            ("15", "14"),
            ("15", "15"),
            ("15", "16"),
            ("15", "17"),
            ("15", "18"),
            ("15", "19"),
            ("15", "20"),
            ("15", "21"),
            ("15", "22"),
            ("15", "24"),
            ("15", "25"),
            ("15", "26"),
            ("15", "27"),
            ("15", "28"),
            ("15", "3"),
            ("15", "4"),
            ("15", "5"),
            ("15", "6"),
            ("15", "7"),
            ("15", "9"),
            ("15", "</s>"),
            ("15", "<s>"),
            ("16", "15"),
            ("17", "15"),
            ("18", "15"),
            ("19", "15"),
            ("20", "15"),
            ("21", "15"),
            ("22", "15"),
            ("23", "15"),
            ("24", "15"),
            ("25", "15"),
            ("25", "17"),
            ("25", "27"),
            ("26", "15"),
            ("27", "15"),
            ("28", "15"),
            ("3", "15"),
            ("4", "1"),
            ("4", "12"),
            ("4", "13"),
            ("4", "15"),
            ("4", "16"),
            ("4", "17"),
            ("4", "19"),
            ("4", "21"),
            ("4", "25"),
            ("4", "27"),
            ("4", "28"),
            ("4", "3"),
            ("4", "4"),
            ("4", "7"),
            ("4", "9"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("5", "15"),
            ("6", "15"),
            ("7", "15"),
            ("8", "15"),
            ("9", "15"),
            ("</s>", "15"),
            ("<s>", "15"),
        }:
            return 11
        elif key in {("15", "2"), ("15", "23"), ("2", "15"), ("25", "2"), ("4", "2")}:
            return 22
        return 16

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 17

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_0_3_output):
        key = (num_attn_1_3_output, num_attn_0_3_output)
        return 2

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 6, 18, 23, 25, 26}:
            return token == "16"
        elif mlp_0_0_output in {1, 3}:
            return token == "14"
        elif mlp_0_0_output in {2}:
            return token == "11"
        elif mlp_0_0_output in {4}:
            return token == "19"
        elif mlp_0_0_output in {24, 5}:
            return token == "15"
        elif mlp_0_0_output in {17, 7}:
            return token == "6"
        elif mlp_0_0_output in {8, 13}:
            return token == "9"
        elif mlp_0_0_output in {9}:
            return token == "</s>"
        elif mlp_0_0_output in {10, 11, 12}:
            return token == "7"
        elif mlp_0_0_output in {14}:
            return token == "8"
        elif mlp_0_0_output in {15, 19, 20, 21, 22, 27, 31}:
            return token == "<pad>"
        elif mlp_0_0_output in {16}:
            return token == "0"
        elif mlp_0_0_output in {28}:
            return token == "18"
        elif mlp_0_0_output in {29}:
            return token == "25"
        elif mlp_0_0_output in {30}:
            return token == "4"

    attn_2_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 16, 18, 20, 21, 22, 23, 25, 26, 30}:
            return mlp_0_1_output == 9
        elif mlp_0_0_output in {1}:
            return mlp_0_1_output == 4
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {3, 4}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {29, 5}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {6}:
            return mlp_0_1_output == 30
        elif mlp_0_0_output in {10, 7}:
            return mlp_0_1_output == 21
        elif mlp_0_0_output in {8, 27}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 20
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 22
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 23
        elif mlp_0_0_output in {13}:
            return mlp_0_1_output == 28
        elif mlp_0_0_output in {14}:
            return mlp_0_1_output == 29
        elif mlp_0_0_output in {15}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {17}:
            return mlp_0_1_output == 27
        elif mlp_0_0_output in {24, 19}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {28}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {31}:
            return mlp_0_1_output == 0

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 20, 22, 23, 25}:
            return token == "16"
        elif mlp_0_0_output in {1}:
            return token == "12"
        elif mlp_0_0_output in {24, 2}:
            return token == "13"
        elif mlp_0_0_output in {3}:
            return token == "24"
        elif mlp_0_0_output in {4}:
            return token == "18"
        elif mlp_0_0_output in {5}:
            return token == "26"
        elif mlp_0_0_output in {6}:
            return token == "19"
        elif mlp_0_0_output in {17, 7}:
            return token == "8"
        elif mlp_0_0_output in {8, 29}:
            return token == "4"
        elif mlp_0_0_output in {9, 11, 12, 13, 14}:
            return token == "9"
        elif mlp_0_0_output in {10}:
            return token == "27"
        elif mlp_0_0_output in {15, 16, 18, 19, 21, 26, 28, 30, 31}:
            return token == "<pad>"
        elif mlp_0_0_output in {27}:
            return token == "2"

    attn_2_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {0}:
            return token == "3"
        elif num_mlp_0_1_output in {1, 20}:
            return token == "10"
        elif num_mlp_0_1_output in {2, 27}:
            return token == "0"
        elif num_mlp_0_1_output in {17, 3}:
            return token == "</s>"
        elif num_mlp_0_1_output in {4}:
            return token == "24"
        elif num_mlp_0_1_output in {29, 5}:
            return token == "13"
        elif num_mlp_0_1_output in {8, 6}:
            return token == "14"
        elif num_mlp_0_1_output in {7}:
            return token == "23"
        elif num_mlp_0_1_output in {24, 9}:
            return token == "<s>"
        elif num_mlp_0_1_output in {10}:
            return token == "<pad>"
        elif num_mlp_0_1_output in {11}:
            return token == "27"
        elif num_mlp_0_1_output in {12}:
            return token == "17"
        elif num_mlp_0_1_output in {13, 22}:
            return token == "8"
        elif num_mlp_0_1_output in {16, 14}:
            return token == "9"
        elif num_mlp_0_1_output in {15}:
            return token == "11"
        elif num_mlp_0_1_output in {18, 21, 31}:
            return token == "1"
        elif num_mlp_0_1_output in {19}:
            return token == "15"
        elif num_mlp_0_1_output in {23}:
            return token == "22"
        elif num_mlp_0_1_output in {25}:
            return token == "28"
        elif num_mlp_0_1_output in {26}:
            return token == "12"
        elif num_mlp_0_1_output in {28, 30}:
            return token == "6"

    attn_2_3_pattern = select_closest(tokens, num_mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            25,
            26,
            31,
        }:
            return token == "10"
        elif mlp_0_0_output in {1}:
            return token == "13"
        elif mlp_0_0_output in {5}:
            return token == "0"
        elif mlp_0_0_output in {9}:
            return token == "17"
        elif mlp_0_0_output in {10, 11}:
            return token == "22"
        elif mlp_0_0_output in {12, 13, 14}:
            return token == "<pad>"
        elif mlp_0_0_output in {28, 29, 30, 15}:
            return token == "15"
        elif mlp_0_0_output in {24, 18, 27}:
            return token == "12"

    num_attn_2_0_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, ones)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 9, 10, 11, 22, 23, 28}:
            return token == "17"
        elif position in {24, 1}:
            return token == "10"
        elif position in {2, 3, 4, 5, 6}:
            return token == "1"
        elif position in {8, 7}:
            return token == "14"
        elif position in {26, 12}:
            return token == "2"
        elif position in {13}:
            return token == "22"
        elif position in {14}:
            return token == "<pad>"
        elif position in {27, 15}:
            return token == "16"
        elif position in {16, 18, 19, 20, 21, 25, 30, 31}:
            return token == "18"
        elif position in {17, 29}:
            return token == "19"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, mlp_0_0_output):
        if position in {0, 3, 4, 5, 6, 7, 8, 9, 15, 16, 18, 19, 20, 21, 22, 24, 30, 31}:
            return mlp_0_0_output == 17
        elif position in {1, 2}:
            return mlp_0_0_output == 2
        elif position in {10, 14}:
            return mlp_0_0_output == 3
        elif position in {17, 11}:
            return mlp_0_0_output == 13
        elif position in {12}:
            return mlp_0_0_output == 9
        elif position in {13}:
            return mlp_0_0_output == 14
        elif position in {23}:
            return mlp_0_0_output == 15
        elif position in {25, 26, 27, 28}:
            return mlp_0_0_output == 8
        elif position in {29}:
            return mlp_0_0_output == 12

    num_attn_2_2_pattern = select(mlp_0_0_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_0_output, token):
        if mlp_1_0_output in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            15,
            16,
            18,
            19,
            20,
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
            return token == "11"
        elif mlp_1_0_output in {24, 1, 21}:
            return token == "13"
        elif mlp_1_0_output in {8}:
            return token == "12"
        elif mlp_1_0_output in {12, 13}:
            return token == "1"
        elif mlp_1_0_output in {14}:
            return token == "14"
        elif mlp_1_0_output in {17}:
            return token == "15"

    num_attn_2_3_pattern = select(tokens, mlp_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_0_output, attn_2_1_output):
        key = (mlp_0_0_output, attn_2_1_output)
        if key in {
            (0, "2"),
            (0, "20"),
            (0, "21"),
            (0, "25"),
            (0, "26"),
            (0, "28"),
            (0, "4"),
            (0, "5"),
            (0, "6"),
            (0, "7"),
            (0, "8"),
            (0, "9"),
            (0, "</s>"),
            (0, "<s>"),
            (4, "4"),
            (4, "6"),
            (4, "9"),
            (5, "4"),
            (5, "6"),
            (5, "9"),
            (7, "21"),
            (7, "23"),
            (7, "24"),
            (7, "25"),
            (7, "27"),
            (7, "3"),
            (7, "4"),
            (7, "6"),
            (7, "7"),
            (7, "8"),
            (7, "9"),
            (7, "</s>"),
            (7, "<s>"),
            (8, "5"),
            (8, "6"),
            (8, "<s>"),
            (9, "18"),
            (9, "19"),
            (9, "28"),
            (9, "5"),
            (9, "6"),
            (9, "9"),
            (9, "<s>"),
            (10, "2"),
            (10, "20"),
            (10, "21"),
            (10, "22"),
            (10, "23"),
            (10, "24"),
            (10, "25"),
            (10, "26"),
            (10, "27"),
            (10, "28"),
            (10, "3"),
            (10, "4"),
            (10, "5"),
            (10, "6"),
            (10, "7"),
            (10, "8"),
            (10, "9"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "2"),
            (11, "20"),
            (11, "21"),
            (11, "23"),
            (11, "24"),
            (11, "25"),
            (11, "26"),
            (11, "27"),
            (11, "28"),
            (11, "4"),
            (11, "5"),
            (11, "6"),
            (11, "7"),
            (11, "8"),
            (11, "9"),
            (11, "</s>"),
            (11, "<s>"),
            (12, "2"),
            (12, "25"),
            (12, "26"),
            (12, "4"),
            (12, "5"),
            (12, "6"),
            (12, "7"),
            (12, "8"),
            (12, "9"),
            (12, "</s>"),
            (12, "<s>"),
            (13, "2"),
            (13, "20"),
            (13, "21"),
            (13, "23"),
            (13, "24"),
            (13, "25"),
            (13, "26"),
            (13, "27"),
            (13, "28"),
            (13, "3"),
            (13, "4"),
            (13, "5"),
            (13, "6"),
            (13, "7"),
            (13, "8"),
            (13, "9"),
            (13, "</s>"),
            (13, "<s>"),
            (14, "2"),
            (14, "20"),
            (14, "21"),
            (14, "23"),
            (14, "24"),
            (14, "25"),
            (14, "26"),
            (14, "27"),
            (14, "4"),
            (14, "6"),
            (14, "7"),
            (14, "8"),
            (14, "9"),
            (14, "</s>"),
            (14, "<s>"),
            (15, "21"),
            (15, "24"),
            (15, "25"),
            (15, "4"),
            (15, "6"),
            (15, "7"),
            (15, "8"),
            (15, "9"),
            (15, "</s>"),
            (15, "<s>"),
            (16, "21"),
            (16, "23"),
            (16, "24"),
            (16, "25"),
            (16, "27"),
            (16, "4"),
            (16, "6"),
            (16, "7"),
            (16, "8"),
            (16, "9"),
            (16, "</s>"),
            (16, "<s>"),
            (18, "25"),
            (18, "26"),
            (18, "4"),
            (18, "6"),
            (18, "7"),
            (18, "8"),
            (18, "9"),
            (18, "</s>"),
            (18, "<s>"),
            (19, "25"),
            (19, "27"),
            (19, "3"),
            (19, "6"),
            (19, "7"),
            (19, "8"),
            (19, "<s>"),
            (20, "21"),
            (20, "25"),
            (20, "26"),
            (20, "27"),
            (20, "4"),
            (20, "6"),
            (20, "7"),
            (20, "8"),
            (20, "</s>"),
            (20, "<s>"),
            (21, "20"),
            (21, "21"),
            (21, "25"),
            (21, "26"),
            (21, "27"),
            (21, "6"),
            (21, "7"),
            (21, "8"),
            (21, "</s>"),
            (21, "<s>"),
            (22, "6"),
            (23, "2"),
            (23, "20"),
            (23, "21"),
            (23, "24"),
            (23, "25"),
            (23, "26"),
            (23, "27"),
            (23, "4"),
            (23, "5"),
            (23, "6"),
            (23, "7"),
            (23, "8"),
            (23, "9"),
            (23, "</s>"),
            (23, "<s>"),
            (24, "20"),
            (24, "21"),
            (24, "25"),
            (24, "26"),
            (24, "27"),
            (24, "6"),
            (24, "7"),
            (24, "8"),
            (24, "</s>"),
            (24, "<s>"),
            (25, "4"),
            (25, "6"),
            (25, "7"),
            (25, "8"),
            (25, "9"),
            (25, "<s>"),
            (26, "28"),
            (26, "5"),
            (26, "6"),
            (26, "7"),
            (26, "8"),
            (26, "9"),
            (26, "<s>"),
            (27, "17"),
            (27, "19"),
            (27, "21"),
            (27, "22"),
            (27, "28"),
            (27, "5"),
            (27, "6"),
            (27, "<s>"),
            (28, "19"),
            (28, "20"),
            (28, "21"),
            (28, "25"),
            (28, "26"),
            (28, "27"),
            (28, "28"),
            (28, "3"),
            (28, "5"),
            (28, "6"),
            (28, "7"),
            (28, "8"),
            (28, "9"),
            (28, "</s>"),
            (28, "<s>"),
            (29, "20"),
            (29, "21"),
            (29, "23"),
            (29, "25"),
            (29, "26"),
            (29, "6"),
            (29, "7"),
            (29, "8"),
            (29, "</s>"),
            (29, "<s>"),
            (30, "21"),
            (30, "23"),
            (30, "25"),
            (30, "26"),
            (30, "27"),
            (30, "4"),
            (30, "6"),
            (30, "7"),
            (30, "8"),
            (30, "9"),
            (30, "</s>"),
            (30, "<s>"),
            (31, "21"),
            (31, "24"),
            (31, "4"),
            (31, "6"),
            (31, "8"),
            (31, "9"),
            (31, "<s>"),
        }:
            return 29
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "10"),
            (0, "11"),
            (0, "15"),
            (0, "16"),
            (1, "0"),
            (1, "1"),
            (1, "10"),
            (1, "11"),
            (1, "14"),
            (1, "15"),
            (1, "16"),
            (1, "17"),
            (1, "18"),
            (1, "19"),
            (1, "2"),
            (1, "20"),
            (1, "21"),
            (1, "22"),
            (1, "23"),
            (1, "24"),
            (1, "25"),
            (1, "26"),
            (1, "27"),
            (1, "28"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "6"),
            (1, "7"),
            (1, "8"),
            (1, "9"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "10"),
            (2, "11"),
            (2, "14"),
            (2, "15"),
            (2, "16"),
            (2, "17"),
            (2, "18"),
            (2, "19"),
            (2, "2"),
            (2, "20"),
            (2, "21"),
            (2, "22"),
            (2, "23"),
            (2, "24"),
            (2, "25"),
            (2, "26"),
            (2, "27"),
            (2, "4"),
            (2, "7"),
            (2, "8"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "10"),
            (3, "11"),
            (3, "14"),
            (3, "15"),
            (3, "16"),
            (3, "17"),
            (3, "18"),
            (3, "2"),
            (3, "20"),
            (3, "23"),
            (3, "24"),
            (3, "25"),
            (3, "27"),
            (3, "7"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "10"),
            (4, "11"),
            (4, "14"),
            (4, "15"),
            (8, "0"),
            (8, "1"),
            (8, "10"),
            (8, "11"),
            (8, "14"),
            (8, "15"),
            (8, "16"),
            (9, "0"),
            (9, "1"),
            (9, "10"),
            (9, "11"),
            (9, "14"),
            (9, "15"),
            (9, "16"),
            (10, "1"),
            (10, "10"),
            (10, "15"),
            (11, "1"),
            (11, "10"),
            (11, "14"),
            (11, "15"),
            (11, "16"),
            (13, "1"),
            (13, "10"),
            (13, "14"),
            (13, "15"),
            (15, "1"),
            (15, "14"),
            (15, "15"),
            (15, "16"),
            (16, "0"),
            (16, "1"),
            (16, "10"),
            (16, "11"),
            (16, "14"),
            (16, "15"),
            (16, "16"),
            (19, "0"),
            (19, "1"),
            (19, "10"),
            (19, "11"),
            (19, "14"),
            (19, "18"),
            (20, "0"),
            (20, "1"),
            (20, "10"),
            (20, "11"),
            (20, "14"),
            (20, "15"),
            (20, "16"),
            (20, "18"),
            (21, "0"),
            (21, "1"),
            (21, "10"),
            (21, "11"),
            (21, "15"),
            (21, "16"),
            (22, "0"),
            (22, "1"),
            (22, "10"),
            (22, "11"),
            (22, "14"),
            (22, "15"),
            (22, "16"),
            (22, "17"),
            (22, "18"),
            (22, "2"),
            (22, "20"),
            (22, "22"),
            (23, "0"),
            (23, "1"),
            (23, "10"),
            (23, "11"),
            (23, "15"),
            (23, "16"),
            (24, "0"),
            (24, "1"),
            (24, "10"),
            (24, "11"),
            (24, "14"),
            (24, "15"),
            (24, "16"),
            (24, "18"),
            (25, "15"),
            (26, "0"),
            (26, "1"),
            (26, "10"),
            (26, "11"),
            (26, "14"),
            (26, "15"),
            (26, "16"),
            (26, "18"),
            (27, "0"),
            (27, "1"),
            (27, "10"),
            (27, "11"),
            (27, "14"),
            (27, "15"),
            (27, "16"),
            (27, "18"),
            (28, "1"),
            (28, "10"),
            (28, "11"),
            (28, "14"),
            (28, "15"),
            (28, "16"),
            (29, "0"),
            (29, "1"),
            (29, "10"),
            (29, "11"),
            (29, "14"),
            (29, "15"),
            (29, "16"),
            (29, "18"),
            (30, "0"),
            (30, "1"),
            (30, "10"),
            (30, "11"),
            (30, "16"),
        }:
            return 10
        elif key in {
            (0, "23"),
            (0, "24"),
            (0, "27"),
            (0, "3"),
            (8, "17"),
            (8, "18"),
            (8, "19"),
            (8, "2"),
            (8, "20"),
            (8, "21"),
            (8, "22"),
            (8, "23"),
            (8, "24"),
            (8, "25"),
            (8, "26"),
            (8, "27"),
            (8, "28"),
            (8, "3"),
            (8, "4"),
            (8, "7"),
            (8, "8"),
            (8, "9"),
            (8, "</s>"),
            (9, "17"),
            (9, "20"),
            (9, "21"),
            (9, "22"),
            (9, "23"),
            (9, "24"),
            (9, "25"),
            (9, "26"),
            (9, "27"),
            (9, "3"),
            (9, "4"),
            (9, "7"),
            (9, "8"),
            (9, "</s>"),
            (17, "0"),
            (17, "1"),
            (17, "10"),
            (17, "11"),
            (17, "14"),
            (17, "15"),
            (17, "16"),
            (17, "17"),
            (17, "18"),
            (17, "19"),
            (17, "2"),
            (17, "20"),
            (17, "21"),
            (17, "22"),
            (17, "23"),
            (17, "24"),
            (17, "25"),
            (17, "26"),
            (17, "27"),
            (17, "28"),
            (17, "3"),
            (17, "4"),
            (17, "5"),
            (17, "6"),
            (17, "7"),
            (17, "8"),
            (17, "9"),
            (17, "</s>"),
            (17, "<s>"),
            (19, "21"),
            (19, "4"),
            (19, "9"),
            (19, "</s>"),
            (20, "3"),
            (20, "9"),
            (21, "23"),
            (21, "24"),
            (21, "4"),
            (21, "9"),
            (24, "4"),
            (24, "9"),
            (26, "27"),
            (26, "3"),
            (26, "4"),
            (26, "</s>"),
            (27, "27"),
            (27, "3"),
            (27, "4"),
            (27, "7"),
            (27, "8"),
            (27, "9"),
            (27, "</s>"),
            (29, "24"),
            (29, "27"),
            (29, "4"),
            (29, "9"),
            (30, "24"),
        }:
            return 9
        elif key in {
            (10, "0"),
            (10, "11"),
            (11, "11"),
            (12, "10"),
            (12, "11"),
            (13, "11"),
            (14, "0"),
            (14, "1"),
            (14, "10"),
            (14, "11"),
            (14, "15"),
            (15, "0"),
            (15, "10"),
            (15, "11"),
            (18, "0"),
            (18, "1"),
            (18, "10"),
            (18, "11"),
            (25, "0"),
            (25, "1"),
            (25, "10"),
            (25, "11"),
            (25, "14"),
            (31, "0"),
            (31, "1"),
            (31, "10"),
            (31, "11"),
            (31, "15"),
        }:
            return 17
        elif key in {
            (2, "9"),
            (3, "4"),
            (3, "9"),
            (19, "23"),
            (19, "24"),
            (19, "26"),
            (22, "21"),
            (22, "23"),
            (22, "24"),
            (22, "25"),
            (22, "26"),
            (22, "27"),
            (22, "4"),
            (22, "7"),
            (22, "8"),
            (22, "9"),
            (22, "</s>"),
            (22, "<s>"),
            (23, "23"),
            (24, "2"),
            (24, "23"),
        }:
            return 7
        elif key in {
            (20, "23"),
            (20, "24"),
            (24, "24"),
            (26, "2"),
            (26, "21"),
            (26, "23"),
            (26, "24"),
            (26, "25"),
            (26, "26"),
            (27, "2"),
            (27, "20"),
            (27, "23"),
            (27, "24"),
            (27, "25"),
            (27, "26"),
            (28, "2"),
            (28, "23"),
            (28, "24"),
            (28, "4"),
        }:
            return 1
        elif key in {
            (0, "14"),
            (4, "16"),
            (10, "14"),
            (12, "14"),
            (12, "15"),
            (21, "14"),
            (23, "14"),
            (30, "14"),
            (30, "15"),
        }:
            return 0
        elif key in {(11, "0"), (12, "0"), (12, "1"), (13, "0"), (28, "0")}:
            return 6
        elif key in {(9, "2"), (19, "2"), (19, "20"), (26, "20"), (26, "22")}:
            return 30
        return 15

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, attn_2_1_output):
        key = (attn_2_3_output, attn_2_1_output)
        if key in {
            ("0", "18"),
            ("1", "18"),
            ("10", "18"),
            ("10", "4"),
            ("11", "18"),
            ("11", "4"),
            ("12", "18"),
            ("13", "18"),
            ("13", "4"),
            ("14", "18"),
            ("14", "4"),
            ("15", "18"),
            ("16", "18"),
            ("16", "4"),
            ("17", "18"),
            ("18", "0"),
            ("18", "1"),
            ("18", "10"),
            ("18", "11"),
            ("18", "12"),
            ("18", "13"),
            ("18", "14"),
            ("18", "15"),
            ("18", "16"),
            ("18", "17"),
            ("18", "18"),
            ("18", "19"),
            ("18", "2"),
            ("18", "20"),
            ("18", "21"),
            ("18", "22"),
            ("18", "23"),
            ("18", "24"),
            ("18", "25"),
            ("18", "26"),
            ("18", "27"),
            ("18", "28"),
            ("18", "3"),
            ("18", "4"),
            ("18", "5"),
            ("18", "6"),
            ("18", "7"),
            ("18", "8"),
            ("18", "9"),
            ("18", "</s>"),
            ("18", "<s>"),
            ("19", "18"),
            ("19", "4"),
            ("2", "18"),
            ("20", "18"),
            ("20", "4"),
            ("21", "18"),
            ("22", "18"),
            ("22", "4"),
            ("23", "18"),
            ("24", "18"),
            ("25", "18"),
            ("25", "4"),
            ("26", "18"),
            ("26", "4"),
            ("27", "18"),
            ("28", "18"),
            ("28", "4"),
            ("3", "18"),
            ("3", "4"),
            ("4", "18"),
            ("4", "4"),
            ("5", "18"),
            ("5", "28"),
            ("5", "4"),
            ("6", "18"),
            ("6", "4"),
            ("7", "12"),
            ("7", "18"),
            ("7", "23"),
            ("7", "4"),
            ("8", "18"),
            ("9", "18"),
            ("9", "4"),
            ("</s>", "18"),
            ("</s>", "4"),
            ("<s>", "18"),
        }:
            return 18
        elif key in {
            ("1", "3"),
            ("11", "3"),
            ("12", "3"),
            ("13", "3"),
            ("14", "</s>"),
            ("15", "28"),
            ("15", "3"),
            ("15", "</s>"),
            ("16", "3"),
            ("17", "3"),
            ("2", "28"),
            ("2", "3"),
            ("2", "</s>"),
            ("20", "3"),
            ("22", "28"),
            ("22", "3"),
            ("22", "</s>"),
            ("23", "3"),
            ("24", "3"),
            ("25", "28"),
            ("25", "3"),
            ("26", "0"),
            ("26", "16"),
            ("26", "19"),
            ("26", "2"),
            ("26", "21"),
            ("26", "22"),
            ("26", "23"),
            ("26", "25"),
            ("26", "26"),
            ("26", "27"),
            ("26", "28"),
            ("26", "3"),
            ("26", "</s>"),
            ("26", "<s>"),
            ("27", "0"),
            ("27", "1"),
            ("27", "11"),
            ("27", "12"),
            ("27", "13"),
            ("27", "15"),
            ("27", "16"),
            ("27", "17"),
            ("27", "19"),
            ("27", "2"),
            ("27", "20"),
            ("27", "21"),
            ("27", "22"),
            ("27", "23"),
            ("27", "24"),
            ("27", "25"),
            ("27", "26"),
            ("27", "27"),
            ("27", "28"),
            ("27", "3"),
            ("27", "</s>"),
            ("27", "<s>"),
            ("28", "3"),
            ("3", "26"),
            ("3", "28"),
            ("3", "3"),
            ("3", "</s>"),
        }:
            return 10
        elif key in {
            ("7", "10"),
            ("7", "11"),
            ("7", "13"),
            ("7", "16"),
            ("7", "17"),
            ("7", "19"),
            ("7", "28"),
            ("7", "3"),
            ("7", "7"),
            ("7", "9"),
            ("7", "</s>"),
            ("7", "<s>"),
            ("9", "10"),
            ("9", "11"),
            ("9", "12"),
            ("9", "16"),
            ("9", "17"),
            ("9", "19"),
            ("9", "28"),
            ("9", "3"),
            ("9", "7"),
            ("9", "9"),
            ("9", "</s>"),
        }:
            return 5
        return 6

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output):
        key = num_attn_1_1_output
        if key in {0}:
            return 21
        return 14

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output):
        key = num_attn_2_3_output
        return 14

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_3_outputs]
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


print(
    run(
        [
            "<s>",
            "4",
            "7",
            "26",
            "6",
            "16",
            "28",
            "15",
            "26",
            "16",
            "26",
            "20",
            "6",
            "20",
            "</s>",
        ]
    )
)

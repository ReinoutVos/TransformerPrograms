import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output\dyck1\dyck1_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 15

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 9, 12}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 13, 5}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {10, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {11, 15}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 13

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 7

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 5

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 5
        elif q_position in {1, 14, 15}:
            return k_position == 13
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {3, 12}:
            return k_position == 10
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {11, 5}:
            return k_position == 9
        elif q_position in {9, 7}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 11

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 11
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 3

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 7

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 2

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 3}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {4, 7}:
            return k_position == 14
        elif q_position in {9, 5, 14}:
            return k_position == 12
        elif q_position in {8, 6}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {13}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 4

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 7, 8, 9, 11, 12, 13, 15}:
            return token == ""
        elif position in {1, 2, 3, 4, 6, 10}:
            return token == ")"
        elif position in {14}:
            return token == "<s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 2, 6, 8, 9, 10, 12, 14, 15}:
            return token == ""
        elif position in {1, 3, 4, 5, 7}:
            return token == ")"
        elif position in {11, 13}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {9, 2}:
            return k_position == 12
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 14, 15}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 0
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {11, 12, 7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 7

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15}:
            return token == ""
        elif position in {1, 11, 12, 7}:
            return token == "<pad>"
        elif position in {3}:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 2, 4, 6, 8, 10, 12, 14}:
            return token == ""
        elif position in {1, 11, 13, 15}:
            return token == "<s>"
        elif position in {9, 3, 5, 7}:
            return token == ")"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"(", "<s>"}:
            return position == 3
        elif token in {")"}:
            return position == 9

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 7, 9, 10, 11, 12, 13, 14, 15}:
            return token == ""
        elif position in {1, 2, 3, 4, 5, 6, 8}:
            return token == ")"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_4_output, attn_0_1_output):
        key = (attn_0_4_output, attn_0_1_output)
        if key in {("(", "("), ("<s>", "(")}:
            return 1
        return 10

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        if key in {("(", "("), ("(", ")"), ("(", "<s>"), (")", "(")}:
            return 4
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output, num_attn_0_0_output):
        key = (num_attn_0_4_output, num_attn_0_0_output)
        return 10

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_2_output):
        key = (num_attn_0_5_output, num_attn_0_2_output)
        return 10

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, position):
        if attn_0_3_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_3_output in {")"}:
            return position == 11

    attn_1_0_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 9
        elif attn_0_5_output in {"<s>"}:
            return position == 5

    attn_1_2_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 3
        elif attn_0_3_output in {")"}:
            return position == 9
        elif attn_0_3_output in {"<s>"}:
            return position == 5

    attn_1_3_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_6_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 9
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_1_4_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_2_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 1, 13, 7}:
            return position == 5
        elif mlp_0_1_output in {2, 14, 15}:
            return position == 9
        elif mlp_0_1_output in {11, 10, 3}:
            return position == 1
        elif mlp_0_1_output in {4, 6}:
            return position == 11
        elif mlp_0_1_output in {9, 12, 5}:
            return position == 13
        elif mlp_0_1_output in {8}:
            return position == 3

    attn_1_5_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_7_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 1
        elif attn_0_2_output in {")"}:
            return position == 9
        elif attn_0_2_output in {"<s>"}:
            return position == 7

    attn_1_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, mlp_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 5

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_2_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, mlp_0_0_output):
        if position in {0, 9, 4, 7}:
            return mlp_0_0_output == 1
        elif position in {1, 13}:
            return mlp_0_0_output == 14
        elif position in {2}:
            return mlp_0_0_output == 7
        elif position in {3}:
            return mlp_0_0_output == 6
        elif position in {8, 10, 5, 6}:
            return mlp_0_0_output == 11
        elif position in {11, 12, 14, 15}:
            return mlp_0_0_output == 8

    num_attn_1_0_pattern = select(mlp_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {"(", "<s>"}:
            return k_attn_0_2_output == ")"
        elif q_attn_0_2_output in {")"}:
            return k_attn_0_2_output == ""

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_5_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 1}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 4
        elif mlp_0_0_output in {3, 5, 7}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {4}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {10, 13, 6}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {8, 14}:
            return mlp_0_1_output == 9
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 0
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {15}:
            return mlp_0_1_output == 10

    num_attn_1_2_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_0_output, mlp_0_1_output):
        if num_mlp_0_0_output in {0, 1, 3, 4, 6, 8, 9, 10, 11, 12}:
            return mlp_0_1_output == 2
        elif num_mlp_0_0_output in {2, 14, 15}:
            return mlp_0_1_output == 13
        elif num_mlp_0_0_output in {5}:
            return mlp_0_1_output == 9
        elif num_mlp_0_0_output in {7}:
            return mlp_0_1_output == 6
        elif num_mlp_0_0_output in {13}:
            return mlp_0_1_output == 12

    num_attn_1_3_pattern = select(
        mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_0_output, attn_0_1_output):
        if mlp_0_0_output in {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15}:
            return attn_0_1_output == ")"
        elif mlp_0_0_output in {2, 14}:
            return attn_0_1_output == "<s>"
        elif mlp_0_0_output in {11}:
            return attn_0_1_output == ""

    num_attn_1_4_pattern = select(attn_0_1_outputs, mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {15, 7}:
            return mlp_0_1_output == 15

    num_attn_1_5_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0}:
            return mlp_0_0_output == 0
        elif num_mlp_0_1_output in {1, 10, 6, 14}:
            return mlp_0_0_output == 4
        elif num_mlp_0_1_output in {2}:
            return mlp_0_0_output == 12
        elif num_mlp_0_1_output in {8, 3}:
            return mlp_0_0_output == 3
        elif num_mlp_0_1_output in {4}:
            return mlp_0_0_output == 1
        elif num_mlp_0_1_output in {5, 7, 9, 11, 12}:
            return mlp_0_0_output == 15
        elif num_mlp_0_1_output in {13, 15}:
            return mlp_0_0_output == 6

    num_attn_1_6_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0}:
            return k_mlp_0_1_output == 6
        elif q_mlp_0_1_output in {1, 5}:
            return k_mlp_0_1_output == 4
        elif q_mlp_0_1_output in {2}:
            return k_mlp_0_1_output == 9
        elif q_mlp_0_1_output in {3}:
            return k_mlp_0_1_output == 0
        elif q_mlp_0_1_output in {4}:
            return k_mlp_0_1_output == 11
        elif q_mlp_0_1_output in {6}:
            return k_mlp_0_1_output == 3
        elif q_mlp_0_1_output in {7}:
            return k_mlp_0_1_output == 8
        elif q_mlp_0_1_output in {8}:
            return k_mlp_0_1_output == 12
        elif q_mlp_0_1_output in {9, 11, 12, 15}:
            return k_mlp_0_1_output == 10
        elif q_mlp_0_1_output in {10, 13, 14}:
            return k_mlp_0_1_output == 14

    num_attn_1_7_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_1_2_output):
        key = (attn_1_0_output, attn_1_2_output)
        return 10

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_4_output, attn_1_5_output):
        key = (attn_1_4_output, attn_1_5_output)
        if key in {("(", "("), (")", "("), ("<s>", "(")}:
            return 0
        elif key in {("(", "<s>")}:
            return 9
        return 2

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_4_output):
        key = (num_attn_1_3_output, num_attn_1_4_output)
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output, num_attn_1_2_output):
        key = (num_attn_1_7_output, num_attn_1_2_output)
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_6_output, attn_0_4_output):
        if attn_0_6_output in {"(", ")", "<s>"}:
            return attn_0_4_output == ")"

    attn_2_0_pattern = select_closest(attn_0_4_outputs, attn_0_6_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"(", ")"}:
            return mlp_0_1_output == 2
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_1_output == 4

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, attn_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 15}:
            return token == "("
        elif mlp_0_1_output in {4, 5, 6, 12, 14}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_4_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 5, 8, 9, 14}:
            return position == 7
        elif mlp_0_1_output in {1, 3, 7, 10, 11, 12, 13, 15}:
            return position == 5
        elif mlp_0_1_output in {2}:
            return position == 1
        elif mlp_0_1_output in {4}:
            return position == 9
        elif mlp_0_1_output in {6}:
            return position == 4

    attn_2_3_pattern = select_closest(positions, mlp_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"(", ")", "<s>"}:
            return k_attn_0_6_output == ")"

    attn_2_4_pattern = select_closest(attn_0_6_outputs, attn_0_6_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(num_mlp_1_0_output, attn_0_6_output):
        if num_mlp_1_0_output in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}:
            return attn_0_6_output == ")"
        elif num_mlp_1_0_output in {8}:
            return attn_0_6_output == ""

    attn_2_5_pattern = select_closest(
        attn_0_6_outputs, num_mlp_1_0_outputs, predicate_2_5
    )
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 11
        elif attn_0_2_output in {"<s>"}:
            return position == 2

    attn_2_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_3_output, position):
        if attn_1_3_output in {"("}:
            return position == 5
        elif attn_1_3_output in {")"}:
            return position == 2
        elif attn_1_3_output in {"<s>"}:
            return position == 1

    attn_2_7_pattern = select_closest(positions, attn_1_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {0, 5}:
            return mlp_0_1_output == 5
        elif num_mlp_0_1_output in {1, 4}:
            return mlp_0_1_output == 7
        elif num_mlp_0_1_output in {2, 10, 15}:
            return mlp_0_1_output == 8
        elif num_mlp_0_1_output in {3}:
            return mlp_0_1_output == 0
        elif num_mlp_0_1_output in {8, 6}:
            return mlp_0_1_output == 14
        elif num_mlp_0_1_output in {7}:
            return mlp_0_1_output == 9
        elif num_mlp_0_1_output in {9, 13}:
            return mlp_0_1_output == 13
        elif num_mlp_0_1_output in {11, 12}:
            return mlp_0_1_output == 10
        elif num_mlp_0_1_output in {14}:
            return mlp_0_1_output == 15

    num_attn_2_0_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, mlp_0_0_output):
        if attn_1_3_output in {"("}:
            return mlp_0_0_output == 1
        elif attn_1_3_output in {")"}:
            return mlp_0_0_output == 7
        elif attn_1_3_output in {"<s>"}:
            return mlp_0_0_output == 11

    num_attn_2_1_pattern = select(mlp_0_0_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_6_output, attn_1_3_output):
        if attn_1_6_output in {0, 6}:
            return attn_1_3_output == "<s>"
        elif attn_1_6_output in {1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15}:
            return attn_1_3_output == ")"
        elif attn_1_6_output in {5}:
            return attn_1_3_output == "("
        elif attn_1_6_output in {13}:
            return attn_1_3_output == ""

    num_attn_2_2_pattern = select(attn_1_3_outputs, attn_1_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 1, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15}:
            return token == ""
        elif mlp_0_1_output in {2, 3, 12, 6}:
            return token == "("

    num_attn_2_3_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_7_output, token):
        if attn_1_7_output in {"("}:
            return token == ""
        elif attn_1_7_output in {")"}:
            return token == "("
        elif attn_1_7_output in {"<s>"}:
            return token == "<s>"

    num_attn_2_4_pattern = select(tokens, attn_1_7_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_0_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_4_output, attn_1_4_output):
        if attn_0_4_output in {"(", ")", "<s>"}:
            return attn_1_4_output == ""

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_0_4_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_5_output, attn_1_3_output):
        if attn_1_5_output in {"(", "<s>"}:
            return attn_1_3_output == ")"
        elif attn_1_5_output in {")"}:
            return attn_1_3_output == ""

    num_attn_2_6_pattern = select(attn_1_3_outputs, attn_1_5_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_5_output, mlp_0_0_output):
        if attn_1_5_output in {"(", ")"}:
            return mlp_0_0_output == 8
        elif attn_1_5_output in {"<s>"}:
            return mlp_0_0_output == 14

    num_attn_2_7_pattern = select(mlp_0_0_outputs, attn_1_5_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_2_output):
        key = (attn_2_3_output, attn_2_2_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 6
        elif key in {("(", ")"), (")", "(")}:
            return 9
        return 7

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, attn_2_0_output):
        key = (attn_2_3_output, attn_2_0_output)
        if key in {("(", "("), ("(", "<s>"), ("<s>", "("), ("<s>", "<s>")}:
            return 0
        return 11

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 0

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_4_output, num_attn_1_3_output):
        key = (num_attn_2_4_output, num_attn_1_3_output)
        return 4

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_3_outputs)
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
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
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
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
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
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
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
        ]
    )
)

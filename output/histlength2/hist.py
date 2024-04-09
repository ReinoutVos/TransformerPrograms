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
        "output/histlength2/hist_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)
    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 37
        elif q_position in {1, 23}:
            return k_position == 22
        elif q_position in {24, 2, 10}:
            return k_position == 18
        elif q_position in {25, 3}:
            return k_position == 23
        elif q_position in {35, 4}:
            return k_position == 32
        elif q_position in {20, 5, 36}:
            return k_position == 29
        elif q_position in {33, 6, 8, 42, 15, 16, 19, 22, 26, 28, 31}:
            return k_position == 20
        elif q_position in {7, 41, 45, 13, 17}:
            return k_position == 14
        elif q_position in {38, 9, 34, 30}:
            return k_position == 0
        elif q_position in {18, 11, 12}:
            return k_position == 26
        elif q_position in {37, 40, 43, 46, 14}:
            return k_position == 12
        elif q_position in {32, 27, 44, 21}:
            return k_position == 31
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {39}:
            return k_position == 25
        elif q_position in {47}:
            return k_position == 21

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 11, 16, 21, 22, 27}:
            return k_position == 1
        elif q_position in {1, 33, 3, 4, 5, 35, 38, 8, 9, 15, 17, 20, 31}:
            return k_position == 21
        elif q_position in {2, 7, 41, 42, 28}:
            return k_position == 29
        elif q_position in {34, 37, 6, 12, 30}:
            return k_position == 31
        elif q_position in {39, 40, 10, 13, 24, 25}:
            return k_position == 12
        elif q_position in {26, 14}:
            return k_position == 11
        elif q_position in {18, 43, 47}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 3
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {32}:
            return k_position == 26
        elif q_position in {36}:
            return k_position == 9
        elif q_position in {44}:
            return k_position == 23
        elif q_position in {45}:
            return k_position == 15
        elif q_position in {46}:
            return k_position == 24

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 2, 5}:
            return k_position == 9
        elif q_position in {32, 3, 8, 29, 31}:
            return k_position == 28
        elif q_position in {4}:
            return k_position == 20
        elif q_position in {34, 6, 41, 43, 24, 25}:
            return k_position == 21
        elif q_position in {7, 13, 46, 20, 28}:
            return k_position == 24
        elif q_position in {36, 39, 9, 44, 14, 47, 16, 27}:
            return k_position == 30
        elif q_position in {35, 37, 10, 42, 12, 15, 18}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {17, 22}:
            return k_position == 15
        elif q_position in {38, 19, 23, 26, 30}:
            return k_position == 18
        elif q_position in {40, 45, 21}:
            return k_position == 19
        elif q_position in {33}:
            return k_position == 29

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1", "5"}:
            return k_token == "0"
        elif q_token in {"2", "3"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 30
        elif token in {"1"}:
            return position == 25
        elif token in {"2"}:
            return position == 11
        elif token in {"3"}:
            return position == 22
        elif token in {"4"}:
            return position == 27
        elif token in {"5"}:
            return position == 15
        elif token in {"<s>"}:
            return position == 12

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        return 18

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        return 39

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
        }:
            return 16
        return 21

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 46

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
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
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
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
            "3",
            "0",
            "3",
            "4",
            "1",
            "4",
            "0",
            "2",
            "1",
            "3",
            "1",
            "0",
            "3",
            "4",
            "5",
            "0",
            "4",
            "2",
            "5",
            "0",
            "0",
            "4",
            "2",
            "0",
            "4",
            "5",
            "4",
            "1",
            "4",
            "1",
            "5",
            "2",
            "3",
            "4",
            "5",
            "0",
            "0",
        ]
    )
)

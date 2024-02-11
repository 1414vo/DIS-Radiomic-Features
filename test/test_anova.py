from src.anova import register_factor_interaction_contribution
import math


def test_interaction_calculation():
    explained_sse = {
        1: {(1,): 1.0, (2,): 2.0, (3,): 4.0},
        2: {
            (
                1,
                2,
            ): 3.5,
            (
                1,
                3,
            ): 5.2,
            (
                2,
                3,
            ): 6.5,
        },
        3: {
            (
                1,
                2,
                3,
            ): 9.3
        },
    }
    answer = {
        1: {(1,): 1.0, (2,): 2.0, (3,): 4.0},
        2: {
            (
                1,
                2,
            ): 0.5,
            (
                1,
                3,
            ): 0.2,
            (
                2,
                3,
            ): 0.5,
        },
        3: {
            (
                1,
                2,
                3,
            ): 1.1
        },
    }
    for int_order in explained_sse:
        for comb in explained_sse[int_order]:
            register_factor_interaction_contribution(
                explained_sse, comb, explained_sse[int_order][comb]
            )
            assert math.isclose(explained_sse[int_order][comb], answer[int_order][comb])

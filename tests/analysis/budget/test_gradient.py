import itertools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from deltakit_analyse.error_budget import FittingParameters
from deltakit_analyse.error_budget._gradient import (
    _approximate_derivative_at_point_from_values,
    _variate_ith_parameter_by,
    get_lambda_gradient,
)


def _arr(array: Sequence[float]) -> npt.NDArray[np.floating]:
    return np.asarray(array)


@pytest.mark.parametrize(
    ("parameters", "variations", "i"),
    [
        (_arr([0.0]), _arr([0.0, 1.0, 2.0]), 0),
        (_arr([0.0, 1.0, 2.0]), _arr([0.0, 1.0, 2.0]), 2),
        (np.arange(10), np.arange(5), 8),
    ],
)
def test_variate_ith_parameter_by(
    parameters: npt.NDArray[np.floating],
    variations: npt.NDArray[np.floating],
    i: int,
) -> None:
    res = np.array(list(_variate_ith_parameter_by(parameters, variations, i)))
    assert len(res.shape) == 2
    n, m = res.shape
    assert n == variations.size
    assert m == parameters.size
    all_entries_except_i_mask = np.arange(m) != i
    for array, variation in zip(res, variations, strict=True):
        np.testing.assert_allclose(
            array[all_entries_except_i_mask], parameters[all_entries_except_i_mask]
        )
        assert pytest.approx(variation) == array[i]


@pytest.mark.parametrize(
    ("degree", "func_and_expected_derivative"),
    list(
        itertools.product(
            [3, 4, 5],
            [
                (lambda x: x**2, 1),
                (lambda x: x**3, 3 / 4),
                (lambda x: x**4, 1 / 2),
                (lambda x: 4 * x + 0.5 * x**2, 4.5),
                (lambda x: -30 + 4 * x + 0.5 * x**2, 4.5),
            ],
        )
    ),
)
def test_approximate_derivative_at_point_from_values(
    degree: int,
    func_and_expected_derivative: tuple[
        Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]], float
    ],
) -> None:
    func, expected_derivative = func_and_expected_derivative
    x = np.linspace(0, 1, 100)
    y = func(x)
    gradient_approximation_point = 0.5
    stddevs = 1e-10 + np.zeros_like(x)
    derivative, stddev = _approximate_derivative_at_point_from_values(
        x, y, stddevs, gradient_approximation_point, degree
    )
    assert pytest.approx(expected_derivative) == derivative
    assert pytest.approx(0, abs=1e-7) == stddev


def _create_synthetic_report(
    sweep: npt.NDArray[np.floating],
    rounds_by_distance: dict[int, list[int]],
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    shots = 200_000
    rows: list[dict[str, int | float]] = []
    for noise_parameter in sweep[0]:
        for distance, round_values in rounds_by_distance.items():
            p_round = 0.01 * (noise_parameter / 0.02) ** ((distance + 1) / 4)
            for num_rounds in round_values:
                lep = 0.5 * (1 - (1 - 2 * p_round) ** num_rounds)
                rows.append(
                    {
                        "noise_p": noise_parameter,
                        "distance": distance,
                        "num_rounds": num_rounds,
                        "shots": shots,
                        "fails": int(rng.binomial(shots, lep)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.filterwarnings(
    "ignore:The maximum estimated logical error probability:UserWarning"
)
def test_get_lambda_gradient_with_repeated_noise_parameters() -> None:
    central = np.array([2.5e-3])
    sweep_with_repeats = np.array([[2.5e-3, 1e-3, 1e-3, 6e-3, 6e-3, 15.7e-3, 19.6e-3]])
    sweep_without_repeats = np.array([[2.5e-3, 1e-3, 6e-3, 15.7e-3, 19.6e-3]])
    sweep_with_repeats_different_order = np.array(
        [[2.5e-3, 6e-3, 1e-3, 19.6e-3, 1e-3, 15.7e-3, 6e-3]]
    )
    rounds_by_distance = {3: [8, 16], 5: [8, 16], 7: [8, 16]}
    report = _create_synthetic_report(sweep_with_repeats, rounds_by_distance)

    gradient_with_repeats, stddev_with_repeats = get_lambda_gradient(
        report,
        central,
        sweep_with_repeats,
        ["p"],
        rounds_by_distance,
        FittingParameters(num_points_per_parameters=6),
    )
    gradient_without_repeats, stddev_without_repeats = get_lambda_gradient(
        report,
        central,
        sweep_without_repeats,
        ["p"],
        rounds_by_distance,
        FittingParameters(num_points_per_parameters=4),
    )
    gradient_with_repeats_different_order, stddev_with_repeats_different_order = (
        get_lambda_gradient(
            report,
            central,
            sweep_with_repeats_different_order,
            ["p"],
            rounds_by_distance,
            FittingParameters(num_points_per_parameters=6),
        )
    )

    np.testing.assert_allclose(gradient_with_repeats, gradient_without_repeats)
    np.testing.assert_allclose(stddev_with_repeats, stddev_without_repeats)
    np.testing.assert_allclose(
        gradient_with_repeats, gradient_with_repeats_different_order
    )
    np.testing.assert_allclose(stddev_with_repeats, stddev_with_repeats_different_order)

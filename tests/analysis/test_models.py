"""Independent checks of the shared array-based mathematical models."""

import ast
from pathlib import Path

import numpy as np
import pytest

from deltakit_analyse import numerics
from deltakit_analyse.numerics import (
    logical_error_probability_per_round,
    memory_logical_error_probability,
)


def test_memory_model_matches_odd_parity_probability():
    rounds = np.array([0, 1, 2, 10])
    expected = (1 - 0.98 * 0.96**rounds) / 2
    np.testing.assert_allclose(
        memory_logical_error_probability(rounds, 0.02, spam_error=0.01), expected
    )


def test_memory_model_preserves_tiny_probabilities():
    result = memory_logical_error_probability(10, 1e-20, spam_error=2e-20)
    assert result.shape == ()
    assert result.item() == pytest.approx(1.2e-19, rel=1e-15, abs=0)


def test_memory_model_handles_physical_boundaries_without_warnings():
    with np.errstate(all="raise"):
        result = memory_logical_error_probability([0, 1, 10], 0.5, spam_error=0.01)
        saturated_spam = memory_logical_error_probability(
            [0, 1, 10], 0.5, spam_error=0.5
        )
    np.testing.assert_allclose(result, [0.01, 0.5, 0.5])
    np.testing.assert_array_equal(saturated_spam, [0.5, 0.5, 0.5])


def test_models_broadcast_without_modifying_inputs():
    rounds = np.array([[0], [2], [4]])
    rates = np.array([0.01, 0.02])
    original_rates = rates.copy()
    result = memory_logical_error_probability(rounds, rates)
    assert result.shape == (3, 2)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(rates, original_rates)
    distances = np.array([[3], [5]])
    scaled = logical_error_probability_per_round(distances, [2, 3], amplitude=0.5)
    np.testing.assert_allclose(scaled, [[1 / 8, 1 / 18], [1 / 16, 1 / 54]])


def test_suppression_model_uses_explicit_multiplier_convention():
    result = logical_error_probability_per_round([3, 5, 7], 3, amplitude=1 / 2)
    np.testing.assert_allclose(result, [1 / 18, 1 / 54, 1 / 162])
    assert logical_error_probability_per_round(3, 3, amplitude=0.5).shape == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rounds": -1, "per_round_error": 0.1},
        {"rounds": np.inf, "per_round_error": 0.1},
        {"rounds": 1, "per_round_error": -0.1},
        {"rounds": 1, "per_round_error": 0.6},
        {"rounds": 1, "per_round_error": np.nan},
        {"rounds": 1, "per_round_error": 0.1, "spam_error": -0.1},
        {"rounds": 1, "per_round_error": 0.1, "spam_error": 0.6},
        {"rounds": [1, 2], "per_round_error": [0.1, 0.2, 0.3]},
    ],
)
def test_memory_model_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError, match="must|shape mismatch"):
        memory_logical_error_probability(**kwargs)


@pytest.mark.parametrize(
    ("distances", "suppression", "amplitude"),
    [(0, 2, 1), (-1, 2, 1), (3, 0, 1), (3, -1, 1), (3, 2, -1), (3, np.nan, 1)],
)
def test_suppression_model_rejects_invalid_inputs(distances, suppression, amplitude):
    with pytest.raises(ValueError, match="must"):
        logical_error_probability_per_round(distances, suppression, amplitude=amplitude)


def test_numerical_layer_does_not_import_high_level_dependencies():
    forbidden = {
        "pandas",
        "matplotlib",
        "deltakit_core",
        "deltakit_circuit",
        "deltakit_decode",
    }
    for package_directory in numerics.__path__:
        for source in Path(package_directory).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    assert node.level <= 1, source
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    assert module.split(".")[0] not in forbidden, source
                    if module.split(".")[0] == "deltakit_analyse":
                        assert module.startswith("deltakit_analyse.numerics."), source

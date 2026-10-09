# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Provide functions to perform error-budgeting estimations."""

from ._budget import get_error_budget
from ._discretisation import (
    DiscretisationStrategy,
    get_c_optimal_points,
    get_linear_points,
    get_logarithmic_points,
)
from ._generation import generate_decoder_managers_for_lambda
from ._gradient import inverse_lambda_gradient_at
from ._lambda import inverse_lambda_at, inverse_lambda_interval_at
from ._memory import MemoryGenerator
from ._parameters import FittingParameters, SamplingParameters
from ._post_processing import compute_lambda_and_stddev_from_results
from ._visualisation import plot_error_budget

__all__ = [
    "DiscretisationStrategy",
    "FittingParameters",
    "MemoryGenerator",
    "SamplingParameters",
    "compute_lambda_and_stddev_from_results",
    "generate_decoder_managers_for_lambda",
    "get_c_optimal_points",
    "get_error_budget",
    "get_linear_points",
    "get_logarithmic_points",
    "inverse_lambda_at",
    "inverse_lambda_gradient_at",
    "inverse_lambda_interval_at",
    "plot_error_budget",
]

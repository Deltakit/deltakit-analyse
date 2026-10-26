# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from ._binomial_fit import (
    DEFAULT_MAX_LIKELIHOOD_FACTOR,
    ConfidenceInterval,
    fit_binomial,
    fit_binomial_batch,
    fit_leppr_and_spam,
)
from ._lambda import (
    LambdaData,
    calculate_lambda_and_lambda_stddev,
    calculate_lambda_asymmetric,
)
from ._leppr import (
    LogicalErrorProbabilityPerRoundData,
    calculate_lep_and_lep_stddev,
    calculate_lep_asymmetric,
    compute_logical_error_per_round,
    fit_logical_error_per_round_asymmetric,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from ._quops import (
    predict_distance_for_quops,
    predict_quops_at_distance,
    predict_quops_interval,
)
from .error_budget import (
    DiscretisationStrategy,
    FittingParameters,
    MemoryGenerator,
    SamplingParameters,
    get_error_budget,
    plot_error_budget,
)

# List only public members in `__all__`.
__all__ = [
    "DEFAULT_MAX_LIKELIHOOD_FACTOR",
    "ConfidenceInterval",
    "DiscretisationStrategy",
    "FittingParameters",
    "LambdaData",
    "LogicalErrorProbabilityPerRoundData",
    "MemoryGenerator",
    "SamplingParameters",
    "calculate_lambda_and_lambda_stddev",
    "calculate_lambda_asymmetric",
    "calculate_lep_and_lep_stddev",
    "calculate_lep_asymmetric",
    "compute_logical_error_per_round",
    "fit_binomial",
    "fit_binomial_batch",
    "fit_leppr_and_spam",
    "fit_logical_error_per_round_asymmetric",
    "get_error_budget",
    "plot_error_budget",
    "predict_distance_for_quops",
    "predict_quops_at_distance",
    "predict_quops_interval",
    "simulate_different_round_numbers_for_lep_per_round_estimation",
]

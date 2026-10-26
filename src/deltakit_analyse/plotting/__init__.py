# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Description of ``deltakit.explorer.visualisation`` namespace here."""

from deltakit_analyse.plotting._correlation_matrix import (
    correlation_matrix,
    defect_diagram,
    defect_rates,
)
from deltakit_analyse.plotting._lambda import plot_lambda
from deltakit_analyse.plotting._leppr import plot_leppr
from deltakit_analyse.plotting.plotting import plot
from deltakit_analyse.plotting.results import (
    LambdaResult,
    LogicalErrorProbabilityPerRoundResult,
    interpolate_lambda,
    interpolate_leppr,
)

__all__ = [
    "LambdaResult",
    "LogicalErrorProbabilityPerRoundResult",
    "correlation_matrix",
    "defect_diagram",
    "defect_rates",
    "interpolate_lambda",
    "interpolate_leppr",
    "plot",
    "plot_lambda",
    "plot_leppr",
]

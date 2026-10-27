# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Module explores how the error suppression factor (lambda)
builds a connection between QuOps and code distance.

References
----------
- https://doi.org/10.48550/arXiv.2408.13687
"""

from collections.abc import Callable
from numbers import Integral

from deltakit_analyse._binomial_fit import ConfidenceInterval
from deltakit_analyse._lambda import LambdaData
from deltakit_analyse.numerics.models import (
    logical_error_probability_per_round,
    memory_logical_error_probability,
)


def _equal_or_less_brute_force_search(
    func: Callable[[int], float],
    target: float,
    minimum: int,
    maximum: int,
    step: int,
) -> int | None:
    """Brute force search or the solution at on an interval.

    func(...) is (unfortunately) not always a monotonic function.
    The function searches for smallest D from [min, max],
    with func(D) <= target.

    Args:
        func: A descending function.
        target: A value to search for.
        minimum: Minimum tested value of D.
        maximum: Maximum tested value of D.
        step: Next value to consider is D + step.

    Returns:
        A value satisfying the search, or None.
    """
    while minimum <= maximum:
        value = func(minimum)
        if value <= target:
            return minimum
        minimum += step
    return None


def predict_quops_at_distance(lambda0: float, lambda_: float, distance: int) -> float:
    """Returns the number of QuOps, given distance.

    This uses the definition that the number of QuOps achievable is 1 / pL, where
    pL is the probability of a logical error occurring in distance-D, D-round block.

    Args:
        lambda0: Legacy multiplicative amplitude. For a fitted ``LambdaData``,
            pass ``1 / result.lambda0``, not its denominator directly.
        lambda_: Error suppression factor.
        distance: The distance at which to calculate the number of QuOps.

    Returns:
        The estimated number of QuOps, or infinity when the modelled error
        probability is zero or underflows to zero.
    """
    if (
        not isinstance(distance, Integral)
        or isinstance(distance, bool)
        or distance <= 0
        or distance % 2 == 0
    ):
        msg = (
            "This method requires a positive odd integer distance. "
            f"Distance provided: {distance}"
        )
        raise ValueError(msg)
    lep_per_round = logical_error_probability_per_round(
        distance, lambda_, amplitude=lambda0
    )
    probability = float(memory_logical_error_probability(distance, lep_per_round))
    return float("inf") if probability == 0 else 1.0 / probability


def predict_quops_interval(
    lambda_data: LambdaData, distance: int
) -> ConfidenceInterval:
    """Predict the number of QuOps with a confidence interval.

    QuOps (quantum operations) is the number of logical operations achievable
    before a logical error occurs, taken as ``1 / pL`` for a distance-D, D-round
    block (see :func:`predict_quops_at_distance`). This propagates the asymmetric
    bounds on Λ and Λ₀ from an asymmetric Lambda fit (see
    :func:`calculate_lambda_asymmetric`) into that estimate. The number of QuOps
    grows with both Λ and the fitted denominator Λ₀.

    Args:
        lambda_data: A fit carrying the asymmetric ``lambda_interval`` and
            ``lambda0_interval`` bounds (see :attr:`LambdaData.has_asymmetric_bounds`).
        distance: The odd distance at which to evaluate the number of QuOps.

    Returns:
        The best estimate and its lower and upper bounds.

    Raises:
        ValueError: If ``lambda_data`` does not carry asymmetric bounds.
    """
    if not lambda_data.has_asymmetric_bounds:
        msg = (
            "lambda_data must carry asymmetric bounds. Use "
            "calculate_lambda_asymmetric to produce them."
        )
        raise ValueError(msg)
    # has_asymmetric_bounds guarantees both intervals are set.
    assert lambda_data.lambda_interval is not None
    assert lambda_data.lambda0_interval is not None
    best = predict_quops_at_distance(
        1 / lambda_data.lambda0, lambda_data.lambda_, distance
    )
    low = predict_quops_at_distance(
        1 / lambda_data.lambda0_interval.low,
        lambda_data.lambda_interval.low,
        distance,
    )
    high = predict_quops_at_distance(
        1 / lambda_data.lambda0_interval.high,
        lambda_data.lambda_interval.high,
        distance,
    )
    return ConfidenceInterval(low=low, best=best, high=high)


def predict_distance_for_quops(
    lambda0: float,
    lambda_: float,
    num_quops: float,
    max_distance: int = 999,
) -> int:
    """Returns the nearest odd distance that achieves the desired
    number of QuOps.

    Uses the definition that the number of QuOps achievable at a
    particular distance is 1 / pL, where pL is the probability of a logical error
    occurring during a distance-D, D-round memory experiment
    without state preparation or measurement error.

    Candidate distances whose extrapolated per-round error exceeds 0.5 are
    skipped because they are outside the memory model's physical domain.

    Args:
        lambda0: Legacy multiplicative amplitude, equal to ``1 / result.lambda0``
            for a fitted ``LambdaData`` denominator.
        lambda_: Error suppression factor.
        num_quops: Number of desired QuOps; must be at least 2.
        max_distance: Maximum distance to consider. Defaults to 999.

    Returns:
        The nearest odd distance that achieves the desired number of QuOps.

    Raises:
        ValueError: If no solution is found, ``num_quops < 2``, or ``lambda_ <= 1``.
    """

    if num_quops < 2:
        msg = "Number of QuOps should be at least 2"
        raise ValueError(msg)

    if lambda_ <= 1.0:
        msg = "Lambda should be greater than 1 to ensure error suppression"
        raise ValueError(msg)

    def lep_at_distance(candidate_distance: int) -> float:
        """Evaluate LEP without aborting on unphysical search candidates.

        Small distances can yield LEPPR above ``0.5`` even when larger distances are
        valid. Return infinity for those candidates so they cannot meet the target and
        the search continues.

        Args:
            candidate_distance: Code distance and number of rounds to evaluate.

        Returns:
            LEP for ``candidate_distance``, or infinity if the LEPPR exceeds ``0.5``.
        """
        lep_per_round = logical_error_probability_per_round(
            candidate_distance, lambda_, amplitude=lambda0
        )
        if lep_per_round > 0.5:
            return float("inf")
        return float(
            memory_logical_error_probability(candidate_distance, lep_per_round)
        )

    required_lep = 1.0 / num_quops
    distance = _equal_or_less_brute_force_search(
        lep_at_distance,
        required_lep,
        minimum=1,
        maximum=max_distance,
        step=2,  # seek for odd solutions
    )
    if distance is None:
        text = (
            f"Could not find a solution between [1, {max_distance}] "
            "for LEP(distance) < 1 / QuOps. Increase max_distance."
        )
        raise ValueError(text)
    return distance

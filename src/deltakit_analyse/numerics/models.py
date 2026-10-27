# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Low-level mathematical functions representing the models for LEP (Logical Error
Probability), LEPPR (Logical Error Probability Per Round) and Lambda.

The :pkg:`deltakit_analyse` code often has to compute LEP, LEPPR or Lambda for various
inputs. This module centralises the mathematical models used to compute these quantities
and defines functions that take raw ``numpy`` arrays or array-like inputs and
efficiently compute the corresponding LEP, LEPPR or Lambda.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _finite_array(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return array


def memory_logical_error_probability(
    rounds: npt.ArrayLike,
    per_round_error: npt.ArrayLike,
    *,
    spam_error: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Evaluate logical error probability in a memory experiment from the estimations of
    the SPAM and logical error probability per-round error rates.

    The model is::

        ``(1 - (1 - 2 * spam_error) * (1 - 2 * per_round_error)**rounds) / 2``

    At zero rounds the result is ``spam_error``, including when the per-round error is
    exactly ``0.5``.

    Args:
        rounds: Non-negative round counts. Floating-point values are supported for
            evaluating continuous fit curves.
        per_round_error: Per-round error probabilities in ``[0, 0.5]``.
        spam_error: State preparation and measurement error in ``[0, 0.5]``.

    Returns:
        Estimates of the logical error probability in an array containing floating-point
        values and whose shape is obtained by broadcasting the shapes of ``rounds``,
        ``per_round_error`` and ``spam_error``.
        Scalar inputs produce a zero-dimensional array.

    Raises:
        ValueError: If inputs are non-finite, outside their allowed domains, or cannot
            be broadcast together.
    """
    rounds_array, per_round_array, spam_array = np.broadcast_arrays(
        _finite_array(rounds, "rounds"),
        _finite_array(per_round_error, "per_round_error"),
        _finite_array(spam_error, "spam_error"),
    )
    if np.any(rounds_array < 0):
        msg = "rounds must be non-negative."
        raise ValueError(msg)
    if np.any((per_round_array < 0) | (per_round_array > 0.5)):
        msg = "per_round_error must be within [0, 0.5]."
        raise ValueError(msg)
    if np.any((spam_array < 0) | (spam_array > 0.5)):
        msg = "spam_error must be within [0, 0.5]."
        raise ValueError(msg)

    # log(F) = log(1 - 2*spam) + rounds*log(1 - 2*per_round_error).
    # At error = 0.5, log(0) = -inf is intentional: zero fidelity gives LEP = 0.5.
    with np.errstate(divide="ignore"):
        log_round_fidelity = np.log1p(-2 * per_round_array)
        log_spam_fidelity = np.log1p(-2 * spam_array)
    # 0 rounds should contribute nothing to log(F) above; avoid 0 * -inf and retain SPAM
    # error only.
    log_round_fidelity = np.where(rounds_array == 0, 0.0, log_round_fidelity)
    log_fidelity = rounds_array * log_round_fidelity
    # LEP = (1 - F)/2, we use expm1(log(F)) to avoid cancellation when F is close to 1.
    return np.asarray(-0.5 * np.expm1(log_fidelity + log_spam_fidelity))


def logical_error_probability_per_round(
    distances: npt.ArrayLike,
    suppression_factor: npt.ArrayLike,
    *,
    amplitude: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Predict logical error probability per QEC round (LEPPR) from code distance.

    The distance-scaling model is::

        per_round_error = amplitude / suppression_factor ** ((distance + 1) / 2)

    For example, with ``amplitude=0.5`` and ``suppression_factor=3``, distances ``3``
    and ``5`` give per-round error probabilities of ``1/18`` and ``1/54``. Increasing
    the distance by two divides the error probability by the suppression factor.

    Args:
        distances: Positive code distances at which to evaluate the model.
            Fractional distances are allowed for drawing continuous fit curves.
        suppression_factor: Factor by which the error decreases when distance
            increases by two. Must be positive; values below one describe error
            increasing with distance instead.
        amplitude: Non-negative scale multiplying the distance-dependent term.
            When using a ``LambdaData`` fit, pass ``1 / result.lambda0`` here.

    Returns:
        Per-round logical error probabilities (LEPPR) as a ``float64`` array. NumPy
        broadcasting determines the output shape. Scalar inputs produce a
        zero-dimensional array. Inputs are not modified. Extrapolated values are not
        clipped, so they may exceed the physical probability range.

    Raises:
        ValueError: If inputs are non-finite, outside their domains, or cannot
            be broadcast together.
    """
    distance_array = _finite_array(distances, "distances")
    suppression_array = _finite_array(suppression_factor, "suppression_factor")
    amplitude_array = _finite_array(amplitude, "amplitude")
    if np.any(distance_array <= 0):
        msg = "distances must be positive."
        raise ValueError(msg)
    if np.any(suppression_array <= 0):
        msg = "suppression_factor must be positive."
        raise ValueError(msg)
    if np.any(amplitude_array < 0):
        msg = "amplitude must be non-negative."
        raise ValueError(msg)
    return np.asarray(
        amplitude_array * np.exp(-(distance_array + 1) / 2 * np.log(suppression_array))
    )

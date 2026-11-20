# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Provides utility functions used through deltakit-analyse."""

import numpy as np
import numpy.typing as npt


def _as_integer_array(values: npt.ArrayLike, name: str) -> npt.NDArray[np.int_]:
    """Validate count values and return them as platform integers.

    Integer-valued floats are accepted, but fractional, non-finite, non-numeric, and
    out-of-range values raise ``ValueError``. The input shape is preserved.

    Args:
        values: Scalar or array-like count data.
        name: Input name to include in validation errors.

    Raises:
        ValueError: if at least one entry in ``values`` cannot be safely interpreted as
            an integer (e.g., a floating point value).

    Returns:
        An integer array with the same shape as ``values``.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        msg = f"{name} must contain finite integer values."
        raise ValueError(msg)
    if array.dtype.kind == "f" and (
        not np.all(np.isfinite(array)) or not np.all(array == np.floor(array))
    ):
        msg = f"{name} must contain finite integer values."
        raise ValueError(msg)

    # Checking if the values provided fit in ``np.int_``. There is a slight complication
    # on too_large because the largest value representable by an integer is 2**(n-1) - 1
    # which is not a power of 2, and so might not be exactly representable using
    # floating-point representation, but that bound is casted to a floating-point value
    # to perform the comparison.
    integer_limits = np.iinfo(np.int_)
    too_small = bool(np.any(array < integer_limits.min))
    if array.dtype.kind == "f":
        if np.finfo(array.dtype).max <= integer_limits.max:
            # Easy check, if the floating point type cannot represent values larger than
            # the integer type, we know already that's fine.
            too_large = False
        else:
            rounded_max = array.dtype.type(integer_limits.max)
            upper_exclusive = array.dtype.type(integer_limits.max + 1)
            # If max rounds to the exclusive bound, equality is out of range.
            if rounded_max == upper_exclusive:
                too_large = bool(np.any(array >= upper_exclusive))
            else:
                too_large = bool(np.any(array > rounded_max))
    else:
        too_large = bool(np.any(array > integer_limits.max))
    if too_small or too_large:
        msg = f"{name} must be representable as platform integers."
        raise ValueError(msg)
    return array.astype(np.int_, copy=False)

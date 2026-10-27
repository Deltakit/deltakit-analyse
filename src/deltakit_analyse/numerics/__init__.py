# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Array-based mathematical routines without sampling or rendering concerns.

Scalar inputs produce zero-dimensional arrays; array inputs follow NumPy
broadcasting. Importing this namespace still executes the package initializer
and does not make the distribution's other dependencies optional.
"""

from .models import (
    logical_error_probability_per_round,
    memory_logical_error_probability,
)

__all__ = [
    "logical_error_probability_per_round",
    "memory_logical_error_probability",
]

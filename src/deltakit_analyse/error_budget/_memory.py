# (c) Copyright Riverlane 2020-2026. All rights reserved.
from collections.abc import Mapping
from typing import Protocol

from deltakit_circuit import Circuit


class MemoryGenerator(Protocol):
    def __call__(self, distance: int, num_rounds: int) -> Circuit: ...


class PreComputedMemoryGenerator(MemoryGenerator):
    """A memory generator that uses pre-computed circuits.

    Args:
        circuits: a mapping from distance values to another mapping that maps num_rounds
            values to actual quantum circuits. Will be used as
            ``circuits[distance][num_rounds]``.
    """

    def __init__(self, circuits: Mapping[int, Mapping[int, Circuit]]) -> None:
        super().__init__()
        self._circuits = circuits

    def __call__(self, distance: int, num_rounds: int) -> Circuit:
        if distance not in self._circuits:
            msg = f"No circuit provided for {distance=}."
            raise RuntimeError(msg)
        circuits_of_distance = self._circuits[distance]
        if num_rounds not in circuits_of_distance:
            msg = f"No circuit provided for {num_rounds=}."
            raise RuntimeError(msg)
        return circuits_of_distance[num_rounds]

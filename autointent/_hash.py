"""This module provides functionality for hashing data using the xxhash algorithm."""

import pickle
from typing import Any

import xxhash


class Hasher:
    """
    A class that provides methods for hashing data using xxhash.

    This class supports both a class-level method for generating hashes from
    any given value, as well as an instance-level method for progressively
    updating a hash state with new values.
    """

    def __init__(self) -> None:
        """
        Initialize the Hasher instance and sets up the internal xxhash state.

        This state will be used for progressively hashing values using the
        `update` method and obtaining the final digest using `hexdigest`.
        """
        self._state = xxhash.xxh64()

    @classmethod
    def hash(cls, value: Any) -> str:  # noqa: ANN401
        """
        Generate a hash for the given value using xxhash.

        :param value: The value to be hashed. This can be any Python object.

        :return: The resulting hash digest as a hexadecimal string.
        """
        return xxhash.xxh64(pickle.dumps(value)).hexdigest()

    def update(self, value: Any) -> None:  # noqa: ANN401
        """
        Update the internal hash state with the provided value.

        This method will first hash the type of the value, then hash the value
        itself, and update the internal state accordingly.

        :param value: The value to update the hash state with.
        """
        self._state.update(str(type(value)).encode())
        self._state.update(self.hash(value).encode())

    def hexdigest(self) -> str:
        """
        Return the current hash digest as a hexadecimal string.

        This method should be called after one or more `update` calls to get
        the final hash result.

        :return: The resulting hash digest as a hexadecimal string.
        """
        return self._state.hexdigest()

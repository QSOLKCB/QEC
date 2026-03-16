"""
Ternary message alphabet for decoder research.

States:
  +1  support bit value
   0  undecided / ambiguous / suspended
  -1  oppose bit value

All values are numpy.int8.  No implicit randomness.
"""

from __future__ import annotations

import numpy as np


_VALID_TERNARY = np.array([-1, 0, 1], dtype=np.int8)


def encode_ternary(value: int | float | np.ndarray) -> np.int8 | np.ndarray:
    """Deterministically encode a value into the ternary alphabet {-1, 0, +1}.

    Scalar or array input is mapped via sign:
      positive -> +1, negative -> -1, zero -> 0.

    Parameters
    ----------
    value : int, float, or np.ndarray
        Input value(s) to encode.

    Returns
    -------
    np.int8 or np.ndarray of np.int8
        Ternary-encoded value(s).
    """
    arr = np.asarray(value, dtype=np.float64)
    result = np.sign(arr).astype(np.int8)
    if result.ndim == 0:
        return np.int8(result)
    return result


def decode_ternary(message: np.int8 | np.ndarray) -> np.int8 | np.ndarray:
    """Validate and return a ternary message.

    Ensures all values are in {-1, 0, +1}.

    Parameters
    ----------
    message : np.int8 or np.ndarray
        Ternary message(s) to validate.

    Returns
    -------
    np.int8 or np.ndarray of np.int8
        Validated ternary message(s).

    Raises
    ------
    ValueError
        If any value is not in {-1, 0, +1}.
    """
    arr = np.asarray(message, dtype=np.int8)
    if not np.all(np.isin(arr, _VALID_TERNARY)):
        raise ValueError(
            f"Ternary messages must be in {{-1, 0, +1}}, got invalid values"
        )
    if arr.ndim == 0:
        return np.int8(arr)
    return arr

"""Centralized validation for the pipeline orchestration module.

All validation logic for pipeline inputs is collected here.
Each function is deterministic, side-effect free, and raises
``ValueError`` on invalid input.

Version: v71.0.3
"""

_VALID_MODES = ("single", "sweep")


def validate_mode(mode: str) -> None:
    """Validate that *mode* is a recognized pipeline mode.

    Parameters
    ----------
    mode : str
        Pipeline execution mode.

    Raises
    ------
    ValueError
        If *mode* is not ``"single"`` or ``"sweep"``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid pipeline mode: {mode!r}. "
            f"Must be one of {_VALID_MODES}."
        )


def validate_suites(mode: str, suites: list) -> None:
    """Validate that *suites* is well-formed for the given *mode*.

    Parameters
    ----------
    mode : str
        Pipeline execution mode (``"single"`` or ``"sweep"``).
    suites : list[dict]
        List of suite dicts from ``_run_single_genome_suite``.

    Raises
    ------
    ValueError
        - If *suites* is empty.
        - If *mode* is ``"single"`` and *suites* does not contain
          exactly one element.
    """
    if not suites:
        raise ValueError("suites must be a non-empty list.")
    if mode == "single" and len(suites) != 1:
        raise ValueError(
            f"Single mode requires exactly one suite, got {len(suites)}."
        )


def validate_sweep_result(result: dict) -> None:
    """Validate the assembled result dict before stage execution.

    Parameters
    ----------
    result : dict
        The result dict assembled from *suites* and *mode*.

    Raises
    ------
    ValueError
        - If ``"mode"`` key is missing.
        - If mode is ``"sweep"`` and ``"results"`` key is missing
          or is not a list.
        - If mode is ``"single"`` and ``"scenarios"`` key is missing.
    """
    if "mode" not in result:
        raise ValueError("Result dict missing 'mode' key.")

    mode = result["mode"]

    if mode == "sweep":
        if "results" not in result:
            raise ValueError(
                "Sweep result dict missing 'results' key."
            )
        if not isinstance(result["results"], list):
            raise ValueError(
                "'results' must be a list of suite dicts."
            )
    elif mode == "single":
        if "scenarios" not in result:
            raise ValueError(
                "Single result dict missing 'scenarios' key."
            )

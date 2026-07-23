"""Source-bound industry observations kept outside simulated curves."""

from __future__ import annotations


def published_evidence_rows() -> list[dict[str, str]]:
    common = {"comparability": "external_result_not_common_model_benchmark"}
    return [
        {
            **common,
            "record_id": "google_surface_d7_2025",
            "platform": "Google Willow",
            "code": "distance-7 surface-code memory",
            "metric": "logical_error_per_cycle",
            "value": "0.00143",
            "uncertainty": "0.00003",
            "context": "101-qubit experiment; real-time decoder",
            "source_url": "https://www.nature.com/articles/s41586-024-08449-y",
        },
        {
            **common,
            "record_id": "quantinuum_steane_2024",
            "platform": "Quantinuum H2",
            "code": "[[7,1,3]]",
            "metric": "physical_to_logical_error_reduction_range",
            "value": "9.8..500",
            "uncertainty": "",
            "context": "logical Bell-state experiment; post-selection dependent",
            "source_url": "https://arxiv.org/abs/2404.02280",
        },
        {
            **common,
            "record_id": "quantinuum_c4c6_2024",
            "platform": "Quantinuum H2",
            "code": "[[12,2,4]] C4/C6-derived",
            "metric": "physical_to_logical_error_reduction_range",
            "value": "4.7..800",
            "uncertainty": "",
            "context": "logical Bell-state experiment; post-selection dependent",
            "source_url": "https://arxiv.org/abs/2404.02280",
        },
        {
            **common,
            "record_id": "gkp_qutrit_2025",
            "platform": "Yale superconducting cavity",
            "code": "single-mode GKP qutrit",
            "metric": "qec_gain",
            "value": "1.82",
            "uncertainty": "0.03",
            "context": "logical versus best physical qutrit decay rate",
            "source_url": "https://www.nature.com/articles/s41586-025-08899-y",
        },
        {
            **common,
            "record_id": "tesseract_code_2024",
            "platform": "Quantinuum trapped ion",
            "code": "[[16,4,4]] subsystem color code",
            "metric": "reported_error_reduction_order",
            "value": "approximately_10x",
            "uncertainty": "",
            "context": "moderate-depth encoded versus unencoded logical circuits",
            "source_url": "https://arxiv.org/abs/2409.04628",
        },
        {
            **common,
            "record_id": "fusion_erasure_threshold_2023",
            "platform": "hardware-agnostic photonic architecture",
            "code": "fusion-based quantum computation",
            "metric": "fusion_measurement_erasure_threshold",
            "value": "0.1198",
            "uncertainty": "",
            "context": "theoretical threshold, not logical error rate",
            "source_url": "https://www.nature.com/articles/s41467-023-36493-1",
        },
        {
            **common,
            "record_id": "fusion_pauli_threshold_2023",
            "platform": "hardware-agnostic photonic architecture",
            "code": "fusion-based quantum computation",
            "metric": "fusion_measurement_pauli_threshold",
            "value": "0.0107",
            "uncertainty": "",
            "context": "theoretical threshold, not logical error rate",
            "source_url": "https://www.nature.com/articles/s41467-023-36493-1",
        },
    ]


def research_watch_rows() -> list[dict[str, str]]:
    return [
        {
            "topic": "spectral_qldpc_absorbing_sets",
            "result": (
                "DFT block diagonalisation, cycle counts, and Fourier "
                "absorbing-set expressions for lifted-product codes"
            ),
            "adoption": "future_code_design_diagnostic",
            "reason": "structural diagnostic; does not replace exact decoding",
            "source_url": "https://arxiv.org/abs/2607.13666",
        },
        {
            "topic": "bivariate_bicycle_qldpc",
            "result": "[[144,12,12]] high-rate memory target",
            "adoption": "radius_bound_only",
            "reason": "no common circuit-level decoder installed",
            "source_url": "https://arxiv.org/abs/2308.07915",
        },
        {
            "topic": "qudit_qldpc",
            "result": "constructions over GF(q), including q=3",
            "adoption": "research_target",
            "reason": "not interchangeable with the finite qutrit codes",
            "source_url": "https://arxiv.org/abs/2510.06495",
        },
        {
            "topic": "maximum_likelihood_decoding",
            "result": "logical-class likelihood is the optimal objective",
            "adoption": "principle_only",
            "reason": "bounded exact decoder retained; no approximate heuristic added",
            "source_url": "https://arxiv.org/abs/2605.17230",
        },
        {
            "topic": "real_time_surface_decoding",
            "result": "550 ns closed loop; 124 ns decoder at distance 3",
            "adoption": "external_latency_reference",
            "reason": "neural hardware result is not a common-model comparator",
            "source_url": "https://arxiv.org/abs/2605.04892",
        },
    ]

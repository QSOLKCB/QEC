from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from .backend_equivalence_replay_receipts import BackendEquivalenceReplayReceipt, validate_backend_equivalence_replay_receipt
from .optimized_qec_benchmark_receipts import OptimizedQECBenchmarkReceipt, validate_optimized_qec_benchmark_receipt
from .optimized_simulation_specs import OptimizedSimulationSpec, validate_optimized_simulation_spec
from .optimized_telemetry_receipts import OptimizedTelemetryReceipt, validate_optimized_telemetry_receipt

_SCHEMA_VERSION = "OPTIMIZED_SIMULATION_REPORT_V1"
_REPORT_MODE = "DETERMINISTIC_OPTIMIZED_SIMULATION_REPORT"
_MAX_REPORT_SECTIONS = 256
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_REPORT_STATUS = {"OPTIMIZED_SIMULATION_REPORT_DRAFT", "OPTIMIZED_SIMULATION_REPORT_PASSED", "OPTIMIZED_SIMULATION_REPORT_FAILED", "OPTIMIZED_SIMULATION_REPORT_BLOCKED"}
_ALLOWED_REPORT_MODE = {"REPLAY_BOUND_SIMULATION_REPORT", "BENCHMARK_BOUND_SIMULATION_REPORT", "TELEMETRY_BOUND_SIMULATION_REPORT", "DETERMINISTIC_SIMULATION_REPORT", "BLOCKED_SIMULATION_REPORT", _REPORT_MODE}

_FORBIDDEN_TOKENS = (
    "simulation executed", "backend executed", "telemetry emitted", "live telemetry collected", "benchmark loop executed",
    "runtime cache active", "automatic optimization", "quantum advantage", "hardware authority"
)

def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def _hash_payload(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()

def _base_payload(x: Any, key: str) -> dict[str, Any]:
    d = x.to_dict(); d.pop(key, None); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")

def _validate_optional_hash(v: str | None) -> None:
    if v is not None: _validate_hash_format(v)

def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    if tuple(getattr(x, field_name) for x in items) != tuple(range(len(items))): raise ValueError("INDEX_ORDER_MISMATCH")

def _validate_report_status_semantics(x: "OptimizedSimulationReport") -> None:
    if x.report_status == "OPTIMIZED_SIMULATION_REPORT_PASSED":
        if not (x.report_summary.report_passed and x.replay_summary.replay_passed and x.benchmark_summary.benchmark_passed and x.telemetry_summary.telemetry_passed): raise ValueError("INVALID_STATUS_SEMANTICS")
        if not x.report_generated_from_canonical_receipts or x.report_contains_runtime_execution or x.report_contains_live_telemetry or x.report_contains_benchmark_execution: raise ValueError("INVALID_STATUS_SEMANTICS")
    if x.report_status == "OPTIMIZED_SIMULATION_REPORT_FAILED" and x.report_summary.report_failure_count < 1: raise ValueError("INVALID_STATUS_SEMANTICS")

def _validate_internal_links(x: "OptimizedSimulationReport") -> None:
    if x.source_optimized_simulation_spec_hash != x.optimization_lineage.optimized_simulation_spec_hash: raise ValueError("LINEAGE_MISMATCH")
    if x.source_backend_equivalence_replay_receipt_hash != x.optimization_lineage.backend_equivalence_replay_receipt_hash: raise ValueError("LINEAGE_MISMATCH")
    if x.source_optimized_qec_benchmark_receipt_hash != x.optimization_lineage.optimized_qec_benchmark_receipt_hash: raise ValueError("LINEAGE_MISMATCH")
    if x.source_optimized_telemetry_receipt_hash != x.optimization_lineage.optimized_telemetry_receipt_hash: raise ValueError("LINEAGE_MISMATCH")

@dataclass(frozen=True)
class SimulationReportSection:
    section_index:int; section_name:str; section_kind:str; section_summary:str; source_hashes:tuple[str,...]; section_status:str; reason:str; simulation_report_section_hash:str
    def to_dict(self)->dict[str,Any]: return {**self.__dict__,"source_hashes":list(self.source_hashes)}

@dataclass(frozen=True)
class SimulationReplaySummary:
    replay_passed:bool; replay_receipt_hash:str; replay_scenario_count:int; replay_observation_count:int; replay_comparison_count:int; replay_failure_count:int; replay_summary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class SimulationBenchmarkSummary:
    benchmark_passed:bool; benchmark_receipt_hash:str; benchmark_measurement_count:int; benchmark_claim_count:int; benchmark_failure_count:int; benchmark_summary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class SimulationTelemetrySummary:
    telemetry_passed:bool; telemetry_receipt_hash:str; telemetry_metric_count:int; telemetry_aggregation_count:int; telemetry_claim_count:int; telemetry_failure_count:int; telemetry_summary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class SimulationOptimizationLineage:
    optimized_simulation_spec_hash:str; backend_equivalence_replay_receipt_hash:str; optimized_qec_benchmark_receipt_hash:str; optimized_telemetry_receipt_hash:str; optimization_scope:str; dependency_name:str; dependency_class:str; lineage_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class SimulationReportSummary:
    report_passed:bool; replay_passed:bool; benchmark_passed:bool; telemetry_passed:bool; report_section_count:int; report_failure_count:int; report_summary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class OptimizedSimulationReport:
    schema_version:str; report_mode:str; report_status:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_optimized_simulation_spec_hash:str; source_backend_equivalence_replay_receipt_hash:str; source_optimized_qec_benchmark_receipt_hash:str; source_optimized_telemetry_receipt_hash:str
    report_sections:tuple[SimulationReportSection,...]; replay_summary:SimulationReplaySummary; benchmark_summary:SimulationBenchmarkSummary; telemetry_summary:SimulationTelemetrySummary
    optimization_lineage:SimulationOptimizationLineage; report_summary:SimulationReportSummary
    report_generated_from_canonical_receipts:bool; report_contains_runtime_execution:bool; report_contains_live_telemetry:bool; report_contains_benchmark_execution:bool
    optimized_simulation_report_hash:str
    def to_dict(self)->dict[str,Any]:
        return {**self.__dict__,"report_sections":[s.to_dict() for s in self.report_sections],"replay_summary":self.replay_summary.to_dict(),"benchmark_summary":self.benchmark_summary.to_dict(),"telemetry_summary":self.telemetry_summary.to_dict(),"optimization_lineage":self.optimization_lineage.to_dict(),"report_summary":self.report_summary.to_dict()}
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())

def _hash_mint(cls: Any, field: str, kwargs: dict[str, Any]) -> Any:
    k=dict(kwargs); k.pop(field,None); x=cls(**{field:"",**k}); return cls(**{**x.__dict__,field:_hash_payload(_base_payload(x,field))})

build_simulation_report_section=lambda **k:_hash_mint(SimulationReportSection,"simulation_report_section_hash",{**k,"source_hashes":tuple(k.get("source_hashes",()))})
build_simulation_replay_summary=lambda **k:_hash_mint(SimulationReplaySummary,"replay_summary_hash",k)
build_simulation_benchmark_summary=lambda **k:_hash_mint(SimulationBenchmarkSummary,"benchmark_summary_hash",k)
build_simulation_telemetry_summary=lambda **k:_hash_mint(SimulationTelemetrySummary,"telemetry_summary_hash",k)
build_simulation_optimization_lineage=lambda **k:_hash_mint(SimulationOptimizationLineage,"lineage_hash",k)
build_simulation_report_summary=lambda **k:_hash_mint(SimulationReportSummary,"report_summary_hash",k)

def build_optimized_simulation_report(**kwargs: Any) -> OptimizedSimulationReport:
    k=dict(kwargs); k.pop("optimized_simulation_report_hash",None); k["report_sections"]=tuple(k.get("report_sections",())); x=OptimizedSimulationReport(optimized_simulation_report_hash="",**k); validate_optimized_simulation_report(x,True); return OptimizedSimulationReport(**{**x.__dict__,"optimized_simulation_report_hash":_hash_payload(_base_payload(x,"optimized_simulation_report_hash"))})

for _n,_cls,_field in (("section",SimulationReportSection,"simulation_report_section_hash"),("replay",SimulationReplaySummary,"replay_summary_hash"),("benchmark",SimulationBenchmarkSummary,"benchmark_summary_hash"),("telemetry",SimulationTelemetrySummary,"telemetry_summary_hash"),("lineage",SimulationOptimizationLineage,"lineage_hash"),("summary",SimulationReportSummary,"report_summary_hash")):
    globals()[f"validate_simulation_{_n if _n!='summary' else 'report_summary'}"] = (lambda cls=_cls,field=_field: (lambda x: (_validate_hash_format(getattr(x,field)), (_hash_payload(_base_payload(x,field))==getattr(x,field)) or (_ for _ in ()).throw(ValueError("HASH_MISMATCH")), True)[-1]))()

def validate_optimized_simulation_report(x: OptimizedSimulationReport, allow_blank_hash: bool=False) -> bool:
    if x.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if x.report_mode not in _ALLOWED_REPORT_MODE: raise ValueError("INVALID_REPORT_MODE")
    if x.report_status not in _ALLOWED_REPORT_STATUS: raise ValueError("INVALID_REPORT_STATUS")
    for h in (x.source_optimized_simulation_spec_hash,x.source_backend_equivalence_replay_receipt_hash,x.source_optimized_qec_benchmark_receipt_hash,x.source_optimized_telemetry_receipt_hash): _validate_hash_format(h)
    if not x.dependency_name or len(x.dependency_name)>_MAX_NAME_LENGTH or not x.dependency_class or len(x.dependency_class)>_MAX_NAME_LENGTH: raise ValueError("INVALID_INPUT")
    if len(x.report_sections)>_MAX_REPORT_SECTIONS: raise ValueError("INVALID_INPUT")
    _validate_dense_indices(x.report_sections,"section_index")
    for s in x.report_sections:
        validate_simulation_section(s)
        if len(s.reason)>_MAX_REASON_LENGTH or any(tok in s.section_summary.lower() for tok in _FORBIDDEN_TOKENS): raise ValueError("FORBIDDEN_CONTENT")
    validate_simulation_replay(x.replay_summary); validate_simulation_benchmark(x.benchmark_summary); validate_simulation_telemetry(x.telemetry_summary); validate_simulation_lineage(x.optimization_lineage); validate_simulation_report_summary(x.report_summary)
    if x.replay_summary.replay_receipt_hash != x.source_backend_equivalence_replay_receipt_hash: raise ValueError("LINK_MISMATCH")
    if x.benchmark_summary.benchmark_receipt_hash != x.source_optimized_qec_benchmark_receipt_hash: raise ValueError("LINK_MISMATCH")
    if x.telemetry_summary.telemetry_receipt_hash != x.source_optimized_telemetry_receipt_hash: raise ValueError("LINK_MISMATCH")
    if x.benchmark_summary.benchmark_passed and not x.replay_summary.replay_passed: raise ValueError("ORDERING_VIOLATION")
    if x.telemetry_summary.telemetry_passed and not x.benchmark_summary.benchmark_passed: raise ValueError("ORDERING_VIOLATION")
    if x.report_summary.report_section_count != len(x.report_sections): raise ValueError("COUNT_MISMATCH")
    rf = sum(1 for v in (x.replay_summary.replay_failure_count,x.benchmark_summary.benchmark_failure_count,x.telemetry_summary.telemetry_failure_count) if v>0)
    if x.report_summary.report_failure_count != rf: raise ValueError("COUNT_MISMATCH")
    if x.report_summary.report_passed != (x.replay_summary.replay_passed and x.benchmark_summary.benchmark_passed and x.telemetry_summary.telemetry_passed): raise ValueError("SUMMARY_MISMATCH")
    _validate_internal_links(x); _validate_report_status_semantics(x)
    exp=_hash_payload(_base_payload(x,"optimized_simulation_report_hash"))
    if x.optimized_simulation_report_hash=="" and allow_blank_hash: return True
    _validate_hash_format(x.optimized_simulation_report_hash)
    if x.optimized_simulation_report_hash!=exp: raise ValueError("HASH_MISMATCH")
    return True


def build_optimized_simulation_report_from_receipts(*, optimized_simulation_spec: OptimizedSimulationSpec, backend_equivalence_replay_receipt: BackendEquivalenceReplayReceipt, optimized_qec_benchmark_receipt: OptimizedQECBenchmarkReceipt, optimized_telemetry_receipt: OptimizedTelemetryReceipt, report_mode: str = "DETERMINISTIC_SIMULATION_REPORT") -> OptimizedSimulationReport:
    validate_optimized_simulation_spec(optimized_simulation_spec); validate_backend_equivalence_replay_receipt(backend_equivalence_replay_receipt); validate_optimized_qec_benchmark_receipt(optimized_qec_benchmark_receipt); validate_optimized_telemetry_receipt(optimized_telemetry_receipt)
    dep_name=optimized_simulation_spec.dependency_name
    if any(x.dependency_name!=dep_name for x in (backend_equivalence_replay_receipt, optimized_qec_benchmark_receipt, optimized_telemetry_receipt)): raise ValueError("DEPENDENCY_MISMATCH")
    dep_class=optimized_simulation_spec.dependency_class; scope=optimized_simulation_spec.optimization_scope
    if any(x.dependency_class!=dep_class for x in (backend_equivalence_replay_receipt, optimized_qec_benchmark_receipt, optimized_telemetry_receipt)): raise ValueError("DEPENDENCY_CLASS_MISMATCH")
    if any(x.optimization_scope!=scope for x in (backend_equivalence_replay_receipt, optimized_qec_benchmark_receipt, optimized_telemetry_receipt)): raise ValueError("OPTIMIZATION_SCOPE_MISMATCH")
    rs=build_simulation_replay_summary(replay_passed=backend_equivalence_replay_receipt.all_comparisons_passed,replay_receipt_hash=backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash,replay_scenario_count=backend_equivalence_replay_receipt.scenario_count,replay_observation_count=backend_equivalence_replay_receipt.observation_count,replay_comparison_count=backend_equivalence_replay_receipt.comparison_result_count,replay_failure_count=0 if backend_equivalence_replay_receipt.all_comparisons_passed else 1)
    bs=build_simulation_benchmark_summary(benchmark_passed=optimized_qec_benchmark_receipt.all_claims_accepted,benchmark_receipt_hash=optimized_qec_benchmark_receipt.optimized_qec_benchmark_receipt_hash,benchmark_measurement_count=optimized_qec_benchmark_receipt.measurement_count,benchmark_claim_count=optimized_qec_benchmark_receipt.claim_count,benchmark_failure_count=0 if optimized_qec_benchmark_receipt.all_claims_accepted else 1)
    ts=build_simulation_telemetry_summary(telemetry_passed=optimized_telemetry_receipt.all_claims_accepted,telemetry_receipt_hash=optimized_telemetry_receipt.optimized_telemetry_receipt_hash,telemetry_metric_count=optimized_telemetry_receipt.metric_count,telemetry_aggregation_count=optimized_telemetry_receipt.aggregation_count,telemetry_claim_count=optimized_telemetry_receipt.claim_count,telemetry_failure_count=0 if optimized_telemetry_receipt.all_claims_accepted else 1)
    sections=(build_simulation_report_section(section_index=0,section_name="replay",section_kind="REPLAY",section_summary="Deterministic replay receipt aggregation.",source_hashes=(rs.replay_receipt_hash,),section_status="PASS" if rs.replay_passed else "FAIL",reason="canonical"),build_simulation_report_section(section_index=1,section_name="benchmark",section_kind="BENCHMARK",section_summary="Bounded benchmark receipt aggregation.",source_hashes=(bs.benchmark_receipt_hash,),section_status="PASS" if bs.benchmark_passed else "FAIL",reason="canonical"),build_simulation_report_section(section_index=2,section_name="telemetry",section_kind="TELEMETRY",section_summary="Replay-safe telemetry receipt aggregation.",source_hashes=(ts.telemetry_receipt_hash,),section_status="PASS" if ts.telemetry_passed else "FAIL",reason="canonical"))
    lineage=build_simulation_optimization_lineage(optimized_simulation_spec_hash=optimized_simulation_spec.optimized_simulation_spec_hash,backend_equivalence_replay_receipt_hash=backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash,optimized_qec_benchmark_receipt_hash=optimized_qec_benchmark_receipt.optimized_qec_benchmark_receipt_hash,optimized_telemetry_receipt_hash=optimized_telemetry_receipt.optimized_telemetry_receipt_hash,optimization_scope=scope,dependency_name=dep_name,dependency_class=dep_class)
    passed=rs.replay_passed and bs.benchmark_passed and ts.telemetry_passed
    summary=build_simulation_report_summary(report_passed=passed,replay_passed=rs.replay_passed,benchmark_passed=bs.benchmark_passed,telemetry_passed=ts.telemetry_passed,report_section_count=3,report_failure_count=sum(int(not v) for v in (rs.replay_passed,bs.benchmark_passed,ts.telemetry_passed)))
    status="OPTIMIZED_SIMULATION_REPORT_PASSED" if passed else "OPTIMIZED_SIMULATION_REPORT_FAILED"
    return build_optimized_simulation_report(schema_version=_SCHEMA_VERSION,report_mode=report_mode,report_status=status,dependency_name=dep_name,dependency_class=dep_class,optimization_scope=scope,source_optimized_simulation_spec_hash=optimized_simulation_spec.optimized_simulation_spec_hash,source_backend_equivalence_replay_receipt_hash=backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash,source_optimized_qec_benchmark_receipt_hash=optimized_qec_benchmark_receipt.optimized_qec_benchmark_receipt_hash,source_optimized_telemetry_receipt_hash=optimized_telemetry_receipt.optimized_telemetry_receipt_hash,report_sections=sections,replay_summary=rs,benchmark_summary=bs,telemetry_summary=ts,optimization_lineage=lineage,report_summary=summary,report_generated_from_canonical_receipts=True,report_contains_runtime_execution=False,report_contains_live_telemetry=False,report_contains_benchmark_execution=False)

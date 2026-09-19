"""Validate a Unity replay report; simulation success never approves hardware.

Written 2026-09-19. Standard library only; no vendor code or robot measurements.
The operator must supply expected asset hashes, mode, duration, maximum sample
gap, and acceptance thresholds. Discrete samples do not guarantee collision-free
motion between samples. The 16 MiB / 100,000 sample limits bound input resources;
they are not physical robot limits or evidence of simulation accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


REPORT_FIELDS = frozenset({
    "schema_version", "scene_id", "model_sha256", "trajectory_sha256",
    "simulator_version", "mode", "completed", "sample_count", "duration_ns",
    "min_clearance_m", "max_head_error_deg", "collision_count",
    "joint_limit_violations", "tracking_lost_count", "samples",
})
SAMPLE_FIELDS = frozenset({
    "time_ns", "clearance_m", "head_error_deg", "collision",
    "joint_limit_violation", "tracking_valid",
})
HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
NUMERIC_TOLERANCE = 1e-9  # Representation tolerance, not a safety margin.
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 100_000


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _hash(value: Any) -> bool:
    return type(value) is str and HASH_PATTERN.fullmatch(value) is not None


def _result(reasons: list[str], derived: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "FAIL" if reasons else "PASS_SIMULATION_ONLY",
        "hardware_approved": False,
        "reasons": reasons,
    }
    if derived is not None:
        result["derived"] = derived
    return result


def validate_report(
    report: Any,
    *,
    model_sha256: str,
    trajectory_sha256: str,
    min_clearance: float,
    max_head_error: float,
    expected_mode: str,
    expected_duration_ns: int,
    max_sample_gap_ns: int,
) -> dict[str, Any]:
    """Recompute every aggregate and enforce explicit simulation acceptance limits.

    Hash equality ties the report to operator-selected model and trajectory bytes.
    It cannot prove a simulator actually loaded those bytes, or validate hardware.
    """
    reasons: list[str] = []
    if type(expected_mode) is not str or expected_mode not in ("head-follow", "dance"):
        reasons.append("expected_mode must be head-follow or dance")
    for label, value in (("expected_duration_ns", expected_duration_ns),
                         ("max_sample_gap_ns", max_sample_gap_ns)):
        if not _integer(value, 1):
            reasons.append(f"{label} must be an explicit positive integer")
    for label, value in (("model_sha256", model_sha256),
                         ("trajectory_sha256", trajectory_sha256)):
        if not _hash(value):
            reasons.append(f"expected {label} must be a 64-digit SHA-256 hex digest")
    for label, value in (("min_clearance", min_clearance),
                         ("max_head_error", max_head_error)):
        if not _finite_number(value) or value < 0:
            reasons.append(f"{label} must be an explicit finite nonnegative number")
    if type(report) is not dict:
        reasons.append("report must be one JSON object")
        return _result(reasons)
    if set(report) != REPORT_FIELDS:
        missing = sorted(REPORT_FIELDS - set(report))
        extra = sorted(str(key) for key in set(report) - REPORT_FIELDS)
        if missing:
            reasons.append("missing report fields: " + ", ".join(missing))
        if extra:
            reasons.append("unknown report fields: " + ", ".join(extra))
        return _result(reasons)
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        reasons.append("schema_version must be integer 1")
    for label in ("scene_id", "simulator_version"):
        if type(report[label]) is not str or not report[label].strip():
            reasons.append(f"{label} must be nonempty text")
    for label, expected in (("model_sha256", model_sha256),
                            ("trajectory_sha256", trajectory_sha256)):
        value = report[label]
        if not _hash(value):
            reasons.append(f"{label} must be a 64-digit SHA-256 hex digest")
        elif _hash(expected) and value.lower() != expected.lower():
            reasons.append(f"{label} does not match the expected asset")
    if report["mode"] not in ("head-follow", "dance"):
        reasons.append("mode must be head-follow or dance")
    elif report["mode"] != expected_mode:
        reasons.append("mode does not match expected_mode")
    if report["completed"] is not True:
        reasons.append("completed must be true")
    for label in ("sample_count", "collision_count", "joint_limit_violations",
                  "tracking_lost_count"):
        if not _integer(report[label]):
            reasons.append(f"{label} must be a nonnegative integer")
    if not _integer(report["duration_ns"], 1):
        reasons.append("duration_ns must be a positive integer")
    for label in ("min_clearance_m", "max_head_error_deg"):
        if not _finite_number(report[label]):
            reasons.append(f"{label} must be a finite number")
    if _finite_number(report["max_head_error_deg"]) and report["max_head_error_deg"] < 0:
        reasons.append("max_head_error_deg must be nonnegative")

    samples = report["samples"]
    if type(samples) is not list or len(samples) < 2:
        reasons.append("samples must contain at least two ordered measurements")
        return _result(reasons)
    if len(samples) > MAX_SAMPLES:
        reasons.append(f"samples exceeds the resource limit of {MAX_SAMPLES}")
        return _result(reasons)
    valid_samples = True
    previous_time: int | None = None
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if type(sample) is not dict or set(sample) != SAMPLE_FIELDS:
            reasons.append(f"{prefix} must contain exactly the sample fields")
            valid_samples = False
            continue
        if not _integer(sample["time_ns"]):
            reasons.append(f"{prefix}.time_ns must be a nonnegative integer")
            valid_samples = False
        else:
            if previous_time is not None and sample["time_ns"] <= previous_time:
                reasons.append(f"{prefix}.time_ns must strictly increase")
            elif (previous_time is not None and _integer(max_sample_gap_ns, 1)
                  and sample["time_ns"] - previous_time > max_sample_gap_ns):
                reasons.append(f"{prefix} gap exceeds max_sample_gap_ns")
            previous_time = sample["time_ns"]
        for label in ("clearance_m", "head_error_deg"):
            if not _finite_number(sample[label]):
                reasons.append(f"{prefix}.{label} must be a finite number")
                valid_samples = False
        if _finite_number(sample["head_error_deg"]) and sample["head_error_deg"] < 0:
            reasons.append(f"{prefix}.head_error_deg must be nonnegative")
        for label in ("collision", "joint_limit_violation", "tracking_valid"):
            if type(sample[label]) is not bool:
                reasons.append(f"{prefix}.{label} must be a boolean")
                valid_samples = False
    if not valid_samples:
        return _result(reasons)

    derived = {
        "sample_count": len(samples),
        "duration_ns": samples[-1]["time_ns"] - samples[0]["time_ns"],
        "min_clearance_m": min(sample["clearance_m"] for sample in samples),
        "max_head_error_deg": max(sample["head_error_deg"] for sample in samples),
        "collision_count": sum(sample["collision"] for sample in samples),
        "joint_limit_violations": sum(sample["joint_limit_violation"] for sample in samples),
        "tracking_lost_count": sum(not sample["tracking_valid"] for sample in samples),
    }
    if (_integer(expected_duration_ns, 1)
            and derived["duration_ns"] != expected_duration_ns):
        reasons.append("sample duration does not match expected_duration_ns")
    for label, observed in derived.items():
        claimed = report[label]
        if label in ("min_clearance_m", "max_head_error_deg"):
            agrees = _finite_number(claimed) and math.isclose(
                claimed, observed, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE)
        else:
            agrees = type(claimed) is int and claimed == observed
        if not agrees:
            reasons.append(f"{label} does not match the supplied samples")
    if _finite_number(min_clearance) and derived["min_clearance_m"] < min_clearance:
        reasons.append("minimum clearance is below the operator threshold")
    if _finite_number(max_head_error) and derived["max_head_error_deg"] > max_head_error:
        reasons.append("maximum head error exceeds the operator threshold")
    if derived["collision_count"]:
        reasons.append("collision observed in the replay")
    if derived["joint_limit_violations"]:
        reasons.append("joint limit violation observed in the replay")
    if report["mode"] == "head-follow" and derived["tracking_lost_count"]:
        reasons.append("head-follow requires valid tracking in every sample")
    return _result(reasons, derived)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def read_report(path: str | Path) -> Any:
    """Read exactly one JSON report; duplicate fields and NaN/Infinity are invalid."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError(f"report exceeds the resource limit of {MAX_REPORT_BYTES} bytes")
    return json.loads(data.decode("utf-8"),
                      object_pairs_hook=_object_without_duplicates,
                      parse_constant=_reject_constant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Unity JSON report file")
    parser.add_argument("--model-sha256", required=True, help="expected model asset SHA-256")
    parser.add_argument("--trajectory-sha256", required=True, help="expected trajectory SHA-256")
    parser.add_argument("--min-clearance", type=float, required=True,
                        help="operator-required minimum clearance in metres")
    parser.add_argument("--max-head-error", type=float, required=True,
                        help="operator-allowed maximum head aiming error in degrees")
    parser.add_argument("--expected-mode", choices=("head-follow", "dance"), required=True,
                        help="mode required for this planned replay")
    parser.add_argument("--expected-duration-ns", type=int, required=True,
                        help="required last-minus-first sample duration in nanoseconds")
    parser.add_argument("--max-sample-gap-ns", type=int, required=True,
                        help="maximum permitted interval between recorded samples")
    args = parser.parse_args(argv)
    try:
        report = read_report(args.report)
        result = validate_report(report, model_sha256=args.model_sha256,
                                 trajectory_sha256=args.trajectory_sha256,
                                 min_clearance=args.min_clearance,
                                 max_head_error=args.max_head_error,
                                 expected_mode=args.expected_mode,
                                 expected_duration_ns=args.expected_duration_ns,
                                 max_sample_gap_ns=args.max_sample_gap_ns)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        result = _result([f"cannot validate report: {exc}"])
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "PASS_SIMULATION_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unity replay acceptance checks, written 2026-09-19; no hardware measurements."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from simulation.replay import MAX_REPORT_BYTES, MAX_SAMPLES, main, read_report, validate_report


MODEL = "a" * 64
TRAJECTORY = "b" * 64


def report_fixture():
    # Synthetic values exercise validation; these are not robot safety limits.
    return {
        "schema_version": 1, "scene_id": "synthetic-test-scene",
        "model_sha256": MODEL, "trajectory_sha256": TRAJECTORY,
        "simulator_version": "test-only", "mode": "head-follow", "completed": True,
        "sample_count": 2, "duration_ns": 1_000_000_000,
        "min_clearance_m": 0.2, "max_head_error_deg": 3.0,
        "collision_count": 0, "joint_limit_violations": 0, "tracking_lost_count": 0,
        "samples": [
            {"time_ns": 10_000_000_000, "clearance_m": 0.3, "head_error_deg": 2.0,
             "collision": False, "joint_limit_violation": False, "tracking_valid": True},
            {"time_ns": 11_000_000_000, "clearance_m": 0.2, "head_error_deg": 3.0,
             "collision": False, "joint_limit_violation": False, "tracking_valid": True},
        ],
    }


def validate(report, **kwargs):
    options = dict(model_sha256=MODEL, trajectory_sha256=TRAJECTORY,
                   min_clearance=0.1, max_head_error=5.0, expected_mode="head-follow",
                   expected_duration_ns=1_000_000_000, max_sample_gap_ns=1_000_000_000)
    options.update(kwargs)
    return validate_report(report, **options)


class ReplayTests(unittest.TestCase):
    def assert_failure(self, report, message=None, **kwargs):
        result = validate(report, **kwargs)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["hardware_approved"])
        self.assertTrue(result["reasons"])
        if message:
            self.assertIn(message, " ".join(result["reasons"]))
        return result

    def test_pass_recomputes_metrics_and_never_approves_hardware(self):
        report = report_fixture()
        before = copy.deepcopy(report)
        result = validate(report)
        self.assertEqual(result["status"], "PASS_SIMULATION_ONLY")
        self.assertFalse(result["hardware_approved"])
        self.assertEqual(result["derived"]["duration_ns"], 1_000_000_000)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(report, before)

    def test_reports_require_every_field_and_reject_hardware_claims(self):
        for field in report_fixture():
            report = report_fixture()
            del report[field]
            with self.subTest(field=field):
                self.assert_failure(report, "missing report fields")
        report = report_fixture()
        report["hardware_approved"] = True
        self.assert_failure(report, "unknown report fields")

    def test_no_vacuous_success(self):
        for samples in ([], [report_fixture()["samples"][0]], None, {}, "samples"):
            report = report_fixture()
            report["samples"] = samples
            self.assert_failure(report, "at least two")
        for report in (None, [], "report", True):
            self.assert_failure(report, "JSON object")

    def test_metadata_validation(self):
        for field, bad in (("schema_version", True), ("schema_version", 2),
                           ("schema_version", 1.0), ("scene_id", " "),
                           ("simulator_version", None), ("mode", "manual"),
                           ("completed", False), ("completed", 1)):
            report = report_fixture()
            report[field] = bad
            with self.subTest(field=field, bad=bad):
                self.assert_failure(report)

    def test_hashes_must_match_expected_assets(self):
        for field in ("model_sha256", "trajectory_sha256"):
            for bad in ("c" * 64, "not-a-hash", None, "g" * 64):
                report = report_fixture()
                report[field] = bad
                self.assert_failure(report, field)
            self.assert_failure(report_fixture(), field, **{field: ""})
        report = report_fixture()
        report["model_sha256"] = MODEL.upper()
        self.assertEqual(validate(report)["status"], "PASS_SIMULATION_ONLY")

    def test_aggregate_tampering_is_detected(self):
        for field in ("sample_count", "duration_ns", "min_clearance_m",
                      "max_head_error_deg", "collision_count", "joint_limit_violations",
                      "tracking_lost_count"):
            report = report_fixture()
            report[field] += 1
            with self.subTest(field=field):
                self.assert_failure(report, field + " does not match")

    def test_integer_fields_do_not_accept_booleans_or_floats(self):
        for field in ("sample_count", "duration_ns", "collision_count",
                      "joint_limit_violations", "tracking_lost_count"):
            for bad in (True, False, 0.0, -1, "0"):
                report = report_fixture()
                report[field] = bad
                with self.subTest(field=field, bad=bad):
                    self.assert_failure(report)
        report = report_fixture()
        report["duration_ns"] = 0
        self.assert_failure(report, "positive integer")

    def test_sample_field_types_are_strict(self):
        for field, bad in (("time_ns", True), ("time_ns", 1.0), ("time_ns", -1),
                           ("clearance_m", True), ("head_error_deg", "0"),
                           ("head_error_deg", -1), ("collision", 0),
                           ("joint_limit_violation", 1), ("tracking_valid", "true")):
            report = report_fixture()
            report["samples"][0][field] = bad
            self.assert_failure(report, "samples[0]." + field)
        for sample in ({}, None, [], {**report_fixture()["samples"][0], "extra": 1}):
            report = report_fixture()
            report["samples"][0] = sample
            self.assert_failure(report, "exactly the sample fields")

    def test_nonfinite_numbers_rejected_in_reports_samples_and_thresholds(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 10 ** 400):
            for field in ("min_clearance_m", "max_head_error_deg"):
                report = report_fixture()
                report[field] = bad
                self.assert_failure(report, "finite")
            for field in ("clearance_m", "head_error_deg"):
                report = report_fixture()
                report["samples"][0][field] = bad
                self.assert_failure(report, "finite")
            for field in ("min_clearance", "max_head_error"):
                self.assert_failure(report_fixture(), "finite", **{field: bad})
        for bad in (-1, True, None):
            self.assert_failure(report_fixture(), **{"min_clearance": bad})

    def test_time_must_increase_and_duration_is_last_minus_first(self):
        for second in (10_000_000_000, 9_000_000_000):
            report = report_fixture()
            report["samples"][1]["time_ns"] = second
            self.assert_failure(report, "strictly increase")
        report = report_fixture()
        report["duration_ns"] = report["samples"][-1]["time_ns"]
        self.assert_failure(report, "duration_ns does not match")

    def test_clearance_and_aiming_use_actual_samples(self):
        self.assert_failure(report_fixture(), "clearance", min_clearance=0.21)
        self.assert_failure(report_fixture(), "head error", max_head_error=2.99)
        self.assertEqual(validate(report_fixture(), min_clearance=0.2,
                                  max_head_error=3.0)["status"], "PASS_SIMULATION_ONLY")

    def test_collision_joint_violation_and_follow_tracking_loss_fail(self):
        for sample_field, aggregate, reason in (
            ("collision", "collision_count", "collision observed"),
            ("joint_limit_violation", "joint_limit_violations", "joint limit violation"),
            ("tracking_valid", "tracking_lost_count", "valid tracking"),
        ):
            report = report_fixture()
            report["samples"][0][sample_field] = sample_field != "tracking_valid"
            report[aggregate] = 1
            self.assert_failure(report, reason)
        report["mode"] = "dance"
        self.assertEqual(validate(report, expected_mode="dance")["status"], "PASS_SIMULATION_ONLY")

    def test_expected_mode_rejects_otherwise_valid_wrong_run(self):
        report = report_fixture()
        report["mode"] = "dance"
        self.assert_failure(report, "mode does not match")
        self.assert_failure(report_fixture(), "mode does not match", expected_mode="dance")
        for invalid in (True, None, "", "any", 0):
            self.assert_failure(report_fixture(), "expected_mode must", expected_mode=invalid)

    def test_controller_follow_maps_to_canonical_head_follow_evidence(self):
        self.assertEqual(validate(report_fixture(), expected_mode="follow")["status"],
                         "PASS_SIMULATION_ONLY")
        report = report_fixture()
        report["mode"] = "follow"
        self.assert_failure(report, "mode must be head-follow or dance", expected_mode="follow")
        report["mode"] = "dance"
        self.assert_failure(report, "mode does not match", expected_mode="follow")

    def test_dance_ignores_aiming_gate_but_preserves_truthful_metrics_and_safety(self):
        report = report_fixture()
        report["mode"] = "dance"
        report["samples"][1]["head_error_deg"] = 170.0
        report["max_head_error_deg"] = 170.0
        self.assertEqual(validate(report, expected_mode="dance")["status"],
                         "PASS_SIMULATION_ONLY")
        self.assert_failure(report, "clearance", expected_mode="dance", min_clearance=0.21)
        report["max_head_error_deg"] = 0.0
        self.assert_failure(report, "does not match", expected_mode="dance")
        report["max_head_error_deg"] = 170.0
        report["samples"][1]["head_error_deg"] = float("nan")
        self.assert_failure(report, "finite", expected_mode="dance")

    def test_controller_follow_alias_retains_aiming_and_tracking_gates(self):
        report = report_fixture()
        self.assert_failure(report, "head error", expected_mode="follow", max_head_error=1.0)
        report["tracking_lost_count"] = 1
        report["samples"][1]["tracking_valid"] = False
        self.assert_failure(report, "valid tracking", expected_mode="follow")

    def test_expected_duration_rejects_self_consistent_shortened_or_longer_runs(self):
        for duration in (500_000_000, 1_500_000_000):
            report = report_fixture()
            report["duration_ns"] = duration
            report["samples"][1]["time_ns"] = report["samples"][0]["time_ns"] + duration
            self.assert_failure(report, "sample duration does not match")

    def test_sparse_endpoint_reports_fail_and_bounded_intervals_pass(self):
        report = report_fixture()
        self.assert_failure(report, "gap exceeds", max_sample_gap_ns=500_000_000)
        middle = {**report["samples"][0], "time_ns": 10_500_000_000}
        report["samples"].insert(1, middle)
        report["sample_count"] = 3
        self.assertEqual(validate(report, max_sample_gap_ns=500_000_000)["status"],
                         "PASS_SIMULATION_ONLY")
        middle["time_ns"] -= 1
        self.assert_failure(report, "gap exceeds", max_sample_gap_ns=500_000_000)

    def test_run_parameters_require_positive_exact_integers(self):
        for name in ("expected_duration_ns", "max_sample_gap_ns"):
            for invalid in (True, False, 0, -1, 1.0, "1", None, float("inf")):
                with self.subTest(name=name, invalid=invalid):
                    self.assert_failure(report_fixture(), name + " must", **{name: invalid})

    def test_sample_resource_limit_rejects_before_traversing(self):
        report = report_fixture()
        report["samples"] = [None] * (MAX_SAMPLES + 1)
        report["sample_count"] = MAX_SAMPLES + 1
        self.assert_failure(report, "resource limit")

    def test_aggregate_tolerance_never_relaxes_acceptance_threshold(self):
        report = report_fixture()
        report["min_clearance_m"] += 5e-10
        self.assertEqual(validate(report)["status"], "PASS_SIMULATION_ONLY")
        self.assert_failure(report, "below", min_clearance=0.2 + 1e-10)
        report["min_clearance_m"] += 1e-6
        self.assert_failure(report, "does not match")


class ReplayFileAndCliTests(unittest.TestCase):
    def test_reader_rejects_duplicate_fields_nonfinite_and_multiple_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}',
                         '{"a":NaN}', '{"a":Infinity}', '{}\n{}\n'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_report(path)

    def run_cli(self, path, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--report", str(path), "--model-sha256", MODEL,
                         "--trajectory-sha256", TRAJECTORY, "--min-clearance", "0.1",
                         "--max-head-error", "5", "--expected-mode", "head-follow",
                         "--expected-duration-ns", "1000000000",
                         "--max-sample-gap-ns", "1000000000", *extra])
        return code, json.loads(output.getvalue())

    def test_cli_exit_status_and_hardware_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            path.write_text(json.dumps(report_fixture()) + "\n", encoding="utf-8")
            code, result = self.run_cli(path)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "PASS_SIMULATION_ONLY")
            self.assertFalse(result["hardware_approved"])
            self.assertEqual(self.run_cli(path, "--expected-mode", "follow")[0], 0)
            code, result = self.run_cli(path, "--model-sha256", "c" * 64)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "FAIL")
            for data in (b"\xff", b"not json", b'{"schema_version":NaN}'):
                path.write_bytes(data)
                self.assertEqual(self.run_cli(path)[0], 1)
            self.assertEqual(self.run_cli(Path(directory) / "missing.json")[0], 1)

    def test_cli_requires_explicit_thresholds(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--report", "unused", "--model-sha256", MODEL,
                  "--trajectory-sha256", TRAJECTORY])
        self.assertEqual(error.exception.code, 2)

    def test_cli_requires_all_planned_run_parameters(self):
        args = ["--report", "unused", "--model-sha256", MODEL,
                "--trajectory-sha256", TRAJECTORY, "--min-clearance", "0.1",
                "--max-head-error", "5", "--expected-mode", "head-follow",
                "--expected-duration-ns", "1000000000", "--max-sample-gap-ns", "1000000000"]
        for missing in ("--expected-mode", "--expected-duration-ns", "--max-sample-gap-ns"):
            index = args.index(missing)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args[:index] + args[index + 2:])
            self.assertEqual(error.exception.code, 2)

    def test_cli_run_mismatch_and_zero_intervals_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report_fixture()), encoding="utf-8")
            for flag, value in (("--expected-mode", "dance"),
                                ("--expected-duration-ns", "500000000"),
                                ("--max-sample-gap-ns", "500000000"),
                                ("--expected-duration-ns", "0"),
                                ("--max-sample-gap-ns", "0")):
                self.assertEqual(self.run_cli(path, flag, value)[0], 1)

    def test_file_resource_limit_rejects_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversize.json"
            with path.open("wb") as stream:
                stream.seek(MAX_REPORT_BYTES)
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "resource limit"):
                read_report(path)
            code, result = self.run_cli(path)
            self.assertEqual(code, 1)
            self.assertIn("resource limit", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()

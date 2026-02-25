from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast


def _run_benchmark(out_dir: Path) -> dict[str, object]:
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "benchmark",
        "--suite",
        "benchmarks",
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], payload)


def test_benchmark_smoke(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    summary = _run_benchmark(out)

    assert summary["overall_pass"] is True
    assert (out / "report.md").exists()

    for drug in ["caffeine", "warfarin", "metoprolol"]:
        assert (out / drug / "overlay.png").exists()
        assert (out / drug / "metrics.json").exists()


def test_benchmark_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "bench_a"
    out_b = tmp_path / "bench_b"
    summary_a = _run_benchmark(out_a)
    summary_b = _run_benchmark(out_b)

    assert summary_a == summary_b


def test_benchmark_run_uses_suite_relative_paths(tmp_path: Path) -> None:
    suite_copy = tmp_path / "bench_copy"
    import shutil

    shutil.copytree("benchmarks", suite_copy)

    cwd = tmp_path / "other"
    cwd.mkdir()

    # Simulate running from a different directory while passing a suite path.
    out = tmp_path / "bench_from_elsewhere"
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "benchmark",
        "--suite",
        str(suite_copy),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=str(cwd))

    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
    assert (out / "report.md").exists()

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast


def _run_benchmark(out_dir: Path, deterministic: bool = True) -> dict[str, object]:
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
    if deterministic:
        cmd.extend(["--deterministic", "--seed", "0"])
    subprocess.run(cmd, check=True)
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], payload)


def test_benchmark_smoke(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    summary = _run_benchmark(out)

    assert summary["overall_pass"] is True
    assert summary["deterministic"] is True
    assert summary["seed"] == 0
    assert summary["solver"]["method"] == "BDF"
    assert (out / "report.md").exists()

    for drug in ["caffeine", "warfarin", "metoprolol"]:
        assert (out / drug / "overlay.png").exists()
        assert (out / drug / "metrics.json").exists()


def test_benchmark_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "bench_a"
    out_b = tmp_path / "bench_b"
    summary_a = _run_benchmark(out_a)
    summary_b = _run_benchmark(out_b)

    summary_a["model_metadata"].pop("timestamp_utc")
    summary_b["model_metadata"].pop("timestamp_utc")
    for item in summary_a["results"]:
        item["model_metadata"].pop("timestamp_utc")
    for item in summary_b["results"]:
        item["model_metadata"].pop("timestamp_utc")
    assert summary_a == summary_b


def test_benchmark_run_uses_suite_relative_paths(tmp_path: Path) -> None:
    import shutil

    # Create a fake repo root with benchmarks + examples side by side
    fake_root = tmp_path / "fake_repo"
    fake_root.mkdir()
    shutil.copytree("benchmarks", fake_root / "benchmarks")
    shutil.copytree("examples", fake_root / "examples")

    cwd = tmp_path / "other"
    cwd.mkdir()

    # Run from a different directory while passing an absolute suite path.
    out = tmp_path / "bench_from_elsewhere"
    cmd = [
        sys.executable,
        "-m",
        "physio_sim.cli",
        "benchmark",
        "--suite",
        str(fake_root / "benchmarks"),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=str(cwd))

    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
    assert (out / "report.md").exists()

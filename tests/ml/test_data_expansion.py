"""Data expansion pipeline tests."""



def test_pkdb_expansion_script_runs():
    """PK-DB expansion script should run without crashing in dry-run mode."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/expand_pkdb_cmax.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

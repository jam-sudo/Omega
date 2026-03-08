#!/usr/bin/env python3
"""Extract hardcoded parameters from phase files in src/omega_pbpk/core/.

Uses AST parsing to find:
1. Numeric assignments at module level (e.g., _DEFAULT_X = 123.0)
2. Default argument values in function signatures (e.g., def f(x: float = 5.0))
3. Numeric literals in dataclass field defaults
4. Dictionary literals with numeric values

Output: YAML or JSON to stdout.

Usage:
    python scripts/extract_phase_params.py                     # all core files, YAML
    python scripts/extract_phase_params.py --format json       # JSON output
    python scripts/extract_phase_params.py --file body.py      # single file
    python scripts/extract_phase_params.py --validate          # validate ranges
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Physiological validation ranges
# ---------------------------------------------------------------------------
VALIDATION_RANGES: dict[str, tuple[float, float, str]] = {
    # pattern: (min, max, unit)
    "volume": (0.001, 50.0, "L"),
    "blood_flow": (0.01, 500.0, "L/h"),
    "flow_fraction": (0.0, 1.0, "fraction of CO"),
    "clearance": (0.001, 200.0, "L/h"),
    "gfr": (10.0, 200.0, "mL/min"),
    "mass": (1.0, 5000.0, "g"),
    "partition_coeff": (0.01, 1000.0, "dimensionless"),
    "rate_constant": (0.001, 100.0, "1/h"),
    "permeability": (1e-12, 1e-2, "cm/s"),
    "surface_area": (1.0, 10000.0, "cm^2"),
    "fraction": (0.0, 1.0, "dimensionless"),
    "time": (0.001, 1000.0, "h"),
}

# Heuristic mapping: variable name patterns -> validation category
NAME_PATTERNS: list[tuple[str, str]] = [
    ("volume", "volume"),
    ("vol", "volume"),
    ("_v_", "volume"),
    ("vd_", "volume"),
    ("blood_flow", "blood_flow"),
    ("flow", "blood_flow"),
    ("_q_", "blood_flow"),
    ("cardiac", "blood_flow"),
    ("frac", "fraction"),
    ("fraction", "fraction"),
    ("gfr", "gfr"),
    ("mass", "mass"),
    ("kp", "partition_coeff"),
    ("partition", "partition_coeff"),
    ("clearance", "clearance"),
    ("clint", "clearance"),
    ("cl_", "clearance"),
    ("rate", "rate_constant"),
    ("ka_", "rate_constant"),
    ("ke_", "rate_constant"),
    ("kmc", "rate_constant"),
    ("transit", "time"),
    ("_h:", "time"),
    ("t_end", "time"),
    ("papp", "permeability"),
    ("surface_area", "surface_area"),
    ("fu_", "fraction"),
    ("f_", "fraction"),
]


def classify_parameter(name: str) -> str | None:
    """Classify a parameter name into a validation category."""
    lower = name.lower()
    for pattern, category in NAME_PATTERNS:
        if pattern in lower:
            return category
    return None


def validate_value(name: str, value: float) -> dict[str, Any] | None:
    """Validate a numeric value against physiological ranges.

    Returns a dict with validation info if the value is outside range,
    None if it passes or if the parameter type is unknown.
    """
    category = classify_parameter(name)
    if category is None or category not in VALIDATION_RANGES:
        return None

    lo, hi, unit = VALIDATION_RANGES[category]
    if lo <= value <= hi:
        return None

    return {
        "parameter": name,
        "value": value,
        "category": category,
        "expected_range": f"[{lo}, {hi}]",
        "unit": unit,
        "status": "OUT_OF_RANGE",
    }


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


def extract_numeric_value(node: ast.expr) -> float | None:
    """Extract a numeric value from an AST expression node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = extract_numeric_value(node.operand)
        if val is not None:
            return -val
    return None


def extract_from_file(filepath: Path) -> dict[str, Any]:
    """Extract parameters from a single Python file using AST parsing.

    Returns a dict with:
        module_constants: {name: value} for top-level numeric assignments
        function_defaults: {func_name: {param: default_value}}
        dict_literals: {name: {key: value}} for dicts with numeric values
        list_literals: {name: [values]} for lists of numbers
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"error": f"Could not read {filepath}"}

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return {"error": f"Syntax error in {filepath}: {e}"}

    result: dict[str, Any] = {
        "file": filepath.name,
        "module_constants": {},
        "function_defaults": {},
        "dict_literals": {},
        "list_literals": {},
    }

    for node in ast.iter_child_nodes(tree):
        # Module-level assignments: X = 123.0
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = extract_numeric_value(node.value)
                    if val is not None:
                        result["module_constants"][target.id] = val
                    # Dict literals
                    elif isinstance(node.value, ast.Dict):
                        d = _extract_dict(node.value)
                        if d:
                            result["dict_literals"][target.id] = d
                    # List/tuple literals of numbers
                    elif isinstance(node.value, (ast.List, ast.Tuple)):
                        nums = _extract_num_list(node.value)
                        if nums:
                            result["list_literals"][target.id] = nums

        # Function definitions: extract default parameter values
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = _extract_function_defaults(node)
            if defaults:
                result["function_defaults"][node.name] = defaults

        # Class definitions: look for methods inside
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defaults = _extract_function_defaults(item)
                    if defaults:
                        key = f"{node.name}.{item.name}"
                        result["function_defaults"][key] = defaults

    # Remove empty sections
    return {k: v for k, v in result.items() if v}


def _extract_dict(node: ast.Dict) -> dict[str, float]:
    """Extract string->numeric entries from a dict literal."""
    out: dict[str, float] = {}
    for key, val in zip(node.keys, node.values):
        if key is None:
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            num = extract_numeric_value(val)
            if num is not None:
                out[key.value] = num
    return out


def _extract_num_list(node: ast.List | ast.Tuple) -> list[float]:
    """Extract a list of numeric values from a List/Tuple node."""
    nums: list[float] = []
    for elt in node.elts:
        val = extract_numeric_value(elt)
        if val is not None:
            nums.append(val)
    # Only return if ALL elements are numeric (to avoid partial extractions)
    if len(nums) == len(node.elts) and nums:
        return nums
    return []


def _extract_function_defaults(node: ast.FunctionDef) -> dict[str, float]:
    """Extract numeric default values from a function's parameters."""
    defaults: dict[str, float] = {}
    args = node.args

    # Positional args with defaults
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, default in enumerate(args.defaults):
        arg_idx = n_args - n_defaults + i
        if arg_idx >= 0 and arg_idx < len(args.args):
            param_name = args.args[arg_idx].arg
            val = extract_numeric_value(default)
            if val is not None:
                defaults[param_name] = val

    # Keyword-only args with defaults
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            val = extract_numeric_value(default)
            if val is not None:
                defaults[arg.arg] = val

    return defaults


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hardcoded parameters from Omega PBPK phase files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "omega_pbpk" / "core",
        help="Directory containing phase .py files",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a single file (name only, within --dir)",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate extracted values against physiological ranges",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics only",
    )
    args = parser.parse_args()

    core_dir = args.dir
    if not core_dir.is_dir():
        print(f"Error: {core_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect files to process
    if args.file:
        files = [core_dir / args.file]
        if not files[0].exists():
            print(f"Error: {files[0]} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        files = sorted(core_dir.glob("*.py"))
        # Exclude __init__.py
        files = [f for f in files if f.name != "__init__.py"]

    # Extract parameters from all files
    all_results: list[dict[str, Any]] = []
    for f in files:
        result = extract_from_file(f)
        if result and "error" not in result:
            all_results.append(result)

    # Validation
    validation_flags: list[dict[str, Any]] = []
    if args.validate:
        for result in all_results:
            fname = result.get("file", "unknown")
            # Check module constants
            for name, val in result.get("module_constants", {}).items():
                flag = validate_value(name, val)
                if flag:
                    flag["file"] = fname
                    validation_flags.append(flag)
            # Check function defaults
            for func_name, params in result.get("function_defaults", {}).items():
                for pname, val in params.items():
                    flag = validate_value(pname, val)
                    if flag:
                        flag["file"] = fname
                        flag["function"] = func_name
                        validation_flags.append(flag)

    # Summary mode
    if args.summary:
        total_constants = sum(
            len(r.get("module_constants", {})) for r in all_results
        )
        total_func_defaults = sum(
            sum(len(v) for v in r.get("function_defaults", {}).values())
            for r in all_results
        )
        total_dicts = sum(
            sum(len(v) for v in r.get("dict_literals", {}).values())
            for r in all_results
        )
        total_lists = sum(
            len(r.get("list_literals", {})) for r in all_results
        )
        print(f"Files processed: {len(all_results)}")
        print(f"Module-level constants: {total_constants}")
        print(f"Function default parameters: {total_func_defaults}")
        print(f"Dict literal entries: {total_dicts}")
        print(f"List/tuple literals: {total_lists}")
        print(f"Total numeric parameters: {total_constants + total_func_defaults + total_dicts}")
        if args.validate:
            print(f"Validation flags: {len(validation_flags)}")
            for flag in validation_flags:
                print(
                    f"  {flag['status']}: {flag.get('file', '?')}::{flag['parameter']} "
                    f"= {flag['value']} (expected {flag['expected_range']} {flag['unit']})"
                )
        return

    # Full output
    output: dict[str, Any] = {
        "extraction_metadata": {
            "source_dir": str(core_dir),
            "files_processed": len(all_results),
            "schema_version": "1.0",
        },
        "files": all_results,
    }
    if args.validate:
        output["validation_flags"] = validation_flags

    if args.format == "json":
        json.dump(output, sys.stdout, indent=2, default=str)
        print()
    else:
        # Simple YAML output (no dependency on pyyaml)
        _print_yaml(output)


def _print_yaml(data: Any, indent: int = 0) -> None:
    """Print a dict/list structure as YAML to stdout (no pyyaml dependency)."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, (dict, list)) and val:
                print(f"{prefix}{key}:")
                _print_yaml(val, indent + 1)
            else:
                print(f"{prefix}{key}: {_yaml_scalar(val)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Print first key on same line as dash
                keys = list(item.keys())
                if keys:
                    first_key = keys[0]
                    first_val = item[first_key]
                    if isinstance(first_val, (dict, list)) and first_val:
                        print(f"{prefix}- {first_key}:")
                        _print_yaml(first_val, indent + 2)
                    else:
                        print(f"{prefix}- {first_key}: {_yaml_scalar(first_val)}")
                    for k in keys[1:]:
                        v = item[k]
                        if isinstance(v, (dict, list)) and v:
                            print(f"{prefix}  {k}:")
                            _print_yaml(v, indent + 2)
                        else:
                            print(f"{prefix}  {k}: {_yaml_scalar(v)}")
            else:
                print(f"{prefix}- {_yaml_scalar(item)}")
    else:
        print(f"{prefix}{_yaml_scalar(data)}")


def _yaml_scalar(val: Any) -> str:
    """Format a scalar value for YAML output."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e15:
            return f"{val:.1f}"
        return str(val)
    if isinstance(val, str):
        if any(c in val for c in ":{}\n[]#&*!|>',\""):
            return f'"{val}"'
        return val
    return str(val)


if __name__ == "__main__":
    main()

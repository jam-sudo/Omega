#!/usr/bin/env python3
"""Build the clinical PK dataset from PK-DB, FDA, OpenFDA, and TDC sources.

Usage:
    python scripts/build_clinical_dataset.py
    python scripts/build_clinical_dataset.py --discover   # also fetch unfiltered PK-DB
    python scripts/build_clinical_dataset.py --no-openfda --no-timecourses  # minimal

Outputs:
    data/ml/clinical/clinical_pk.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clinical PK dataset")
    parser.add_argument("--no-pkdb", action="store_true", help="Skip PK-DB groups/individuals")
    parser.add_argument("--no-fda", action="store_true", help="Skip DailyMed FDA labels")
    parser.add_argument("--no-openfda", action="store_true", help="Skip OpenFDA (api.fda.gov)")
    parser.add_argument("--no-timecourses", action="store_true", help="Skip PK-DB C(t) timecourses")
    parser.add_argument("--no-tdc", action="store_true", help="Skip TDC ADME datasets")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Also fetch all PK-DB data without substance filter (discover new drugs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ml/clinical",
        help="Output directory (default: data/ml/clinical)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    from omega_pbpk.ml.data.datasets import run_clinical_data_pipeline

    print("=" * 60)
    print("Clinical PK Dataset Builder")
    print("=" * 60)

    sources = []
    if not args.no_pkdb:
        sources.append("PK-DB groups/individuals")
    if not args.no_timecourses:
        sources.append("PK-DB timecourses")
    if not args.no_fda:
        sources.append("DailyMed FDA")
    if not args.no_openfda:
        sources.append("OpenFDA")
    if not args.no_tdc:
        sources.append("TDC")
    if args.discover:
        sources.append("PK-DB discovery (unfiltered)")
    print(f"Sources: {', '.join(sources)}")

    dataset = run_clinical_data_pipeline(
        output_dir=args.output_dir,
        use_pkdb=not args.no_pkdb,
        use_fda=not args.no_fda,
        use_openfda=not args.no_openfda,
        use_timecourses=not args.no_timecourses,
        use_tdc=not args.no_tdc,
        discover_all=args.discover,
        seed=args.seed,
    )

    # Print summary statistics
    df = dataset.to_dataframe()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")

    if len(df) > 0:
        print("\nRecords by source:")
        for source, count in df.groupby("source").size().items():
            print(f"  {source}: {count}")

        print("\nRecords by parameter:")
        for param, count in df.groupby("parameter").size().items():
            print(f"  {param}: {count}")

        if "compound" in df.columns:
            compounds_with_smiles = df[df["smiles"].notna()]["compound"].nunique()
            compounds_total = df["compound"].nunique()
            print(f"\nUnique compounds: {compounds_total}")
            print(f"Compounds with SMILES: {compounds_with_smiles}")

        if "split" in df.columns:
            print("\nSplit distribution:")
            for split, count in df.groupby("split").size().items():
                print(f"  {split}: {count}")

        if "qc_flag" in df.columns:
            print("\nQC flags:")
            for flag, count in df.groupby("qc_flag").size().items():
                print(f"  {flag}: {count}")

    print(f"\nOutput: {args.output_dir}/clinical_pk.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Build the full 619-bout, seven-subject, three-region analysis table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUBJECTS = {"1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1"}
EVENTS = {"exp1 sniff", "exp5 sniff"}
METRICS = {"power", "coherence"}
REGIONS = {"mPFC", "MD", "NAc", "mPFC_MD", "mPFC_NAc", "NAc_MD"}
ID_COLUMNS = ["subject", "recording", "event", "partner", "trial", "condition"]
FEATURE_COLUMNS = ID_COLUMNS + ["band", "metric", "region_or_pair"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    selected = []
    for chunk in pd.read_csv(args.source, chunksize=200_000):
        chunk["subject"] = chunk["subject"].astype(str)
        keep = (
            chunk["subject"].isin(SUBJECTS)
            & chunk["event"].isin(EVENTS)
            & chunk["metric"].isin(METRICS)
            & chunk["region_or_pair"].isin(REGIONS)
        )
        selected.append(chunk.loc[keep].copy())

    result = pd.concat(selected, ignore_index=True)
    bouts = result[ID_COLUMNS].drop_duplicates()
    rows_per_bout = result.groupby(ID_COLUMNS, dropna=False).size()

    if len(bouts) != 619:
        raise ValueError(f"Expected 619 eligible bouts, found {len(bouts)}")
    if not rows_per_bout.eq(30).all():
        raise ValueError(
            "Every bout must have 30 power/coherence features; distribution: "
            f"{rows_per_bout.value_counts().sort_index().to_dict()}"
        )
    if result.duplicated(FEATURE_COLUMNS).any():
        raise ValueError("Duplicate feature keys detected within eligible bouts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    counts = bouts.groupby(["subject", "condition", "event"]).size().unstack(fill_value=0)
    print(f"Saved {len(result):,} feature rows representing {len(bouts)} bouts")
    print(counts.to_string())


if __name__ == "__main__":
    main()

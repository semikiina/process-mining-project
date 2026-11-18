"""Prepare EL_IM_PD event log files for import in Disco.

Writes three CSV files to data/disco/:
    EL_Issue.csv               events whose case ('Lead Object Id') is an Issue
    EL_ProtocolDeviation.csv   events whose case is a Protocol Deviation
    EL_IM_PD_enriched.csv      all events, with case attributes from IM_PD added:
                                   CaseType, Study, StudyCountry, StudySite, CaseSeverity

The case id is 'Lead Object Id' (e.g. Issue_49765 / ProtocolDeviation_106751), which
matches IM_PD.ID. The original event-level 'Severity' column is kept unchanged;
'CaseSeverity' is the severity of the Issue / PD as recorded in IM_PD.

Usage (from the project root):
    python scripts/prepare_disco_logs.py
    python scripts/prepare_disco_logs.py --data data --out data/disco
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASE_COL = "Lead Object Id"
CASE_ATTRS = {"Study": "Study", "StudyCountry": "StudyCountry",
              "StudySite": "StudySite", "Severity": "CaseSeverity"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data", type=Path, help="folder with the input CSVs")
    ap.add_argument("--out", default=ROOT / "data" / "disco", type=Path, help="output folder")
    ap.add_argument("--el", default="EL_IM_PD.csv")
    ap.add_argument("--imd", default="IM_PD.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # Read everything as text so values (ids, dates) are written back exactly as they came in.
    el = pd.read_csv(a.data / a.el, dtype=str, keep_default_na=False)
    imd = pd.read_csv(a.data / a.imd, dtype=str, keep_default_na=False,
                      usecols=["ID", *CASE_ATTRS])
    print(f"loaded: {len(el):,} events | {len(imd):,} Issues/PDs")

    case_type = el[CASE_COL].str.extract(r"^(Issue|ProtocolDeviation)_", expand=False)
    if case_type.isna().any():
        print(f"WARNING: {case_type.isna().sum():,} events have an unknown case id prefix (not in any output split)")

    # 1) Split into Issues and Protocol Deviations
    for kind in ("Issue", "ProtocolDeviation"):
        part = el[case_type == kind]
        path = a.out / f"EL_{kind}.csv"
        part.to_csv(path, index=False)
        print(f"wrote {path.name}: {len(part):,} events, {part[CASE_COL].nunique():,} cases")

    # 2) Enriched full log: add study / country / site / severity of the case to every event
    attrs = imd.rename(columns={"ID": CASE_COL, **CASE_ATTRS})[[CASE_COL, *CASE_ATTRS.values()]]
    enriched = el.merge(attrs, on=CASE_COL, how="left", validate="many_to_one")
    enriched.insert(enriched.columns.get_loc(CASE_COL) + 1, "CaseType", case_type.fillna(""))
    missing = ~el[CASE_COL].isin(imd["ID"])
    if missing.any():
        print(f"note: {missing.sum():,} events ({el.loc[missing, CASE_COL].nunique():,} cases) "
              f"have no matching row in {a.imd}; their case attributes are left empty")
    path = a.out / "EL_IM_PD_enriched.csv"
    enriched.to_csv(path, index=False)
    print(f"wrote {path.name}: {len(enriched):,} events, columns added: CaseType, {', '.join(CASE_ATTRS.values())}")


if __name__ == "__main__":
    main()

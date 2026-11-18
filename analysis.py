#!/usr/bin/env python3
"""
Step 1 - profile the four Novo Nordisk CSVs and test how they link together.

Usage (from the folder that holds this script):
    python 01_profile_data.py --dir /path/to/csv/folder

Expected files in --dir (any that are missing are skipped):
    Task.csv, TM.csv, IM_PD.csv, EL_IM_PD.csv
Override a name with e.g. --imd IM_PD_Final.csv --el EL_IM_PD.csv

Needs only pandas. Prints a report and saves it to profile_report.txt in --dir.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 70)


class Tee:
    """Write everything to the console and to a report file."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def h(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load(path):
    return pd.read_csv(path, low_memory=False)


def find_col(df, *names):
    """Case-insensitive column lookup; returns the real column name or None."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def prefix(s):
    return s.astype("string").str.split("_", n=1).str[0]


def suffix(s):
    return s.astype("string").str.split("_", n=1).str[1]


def rate(values, ref):
    v = values.dropna().astype(str)
    return float(v.isin(ref).mean()) if len(v) else 0.0


def show_nulls(df):
    n = df.isna().sum()
    n = n[n > 0]
    if n.empty:
        print("no null values")
    else:
        print((pd.DataFrame({"nulls": n, "pct": (n / len(df) * 100).round(1)})).to_string())


# --------------------------------------------------------------------------
def profile_child(name, df):
    """Task.csv and TM.csv have the same structure."""
    h(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    show_nulls(df)
    print(f"\nID unique: {df['ID'].is_unique} (duplicate IDs: {df['ID'].duplicated().sum():,})")

    iss, pdv = df["IssueObject_ID"], df["ProtocolDeviation_ID"]
    a, b = iss.notna(), pdv.notna()
    print("\nLink columns  (Issue only / PD only / both / neither):",
          f"{(a & ~b).sum():,} / {(~a & b).sum():,} / {(a & b).sum():,} / {(~a & ~b).sum():,}")
    si, sp = suffix(iss), suffix(pdv)
    both = a & b
    print(f"Issue_N and ProtocolDeviation_N carry the same number: {(si[both] == sp[both]).mean():.1%}")
    if "Issue" in df.columns:
        eq = (si.astype("string") == df["Issue"].astype("string"))[a]
        print(f"'Issue' column equals that number: {eq.mean():.2%}  ({(~eq).sum():,} rows differ)")
    print(f"Distinct case numbers: {si.nunique():,}; rows per case: median "
          f"{si.value_counts().median():.0f}, max {si.value_counts().max()}")

    created = pd.to_datetime(df["CreatedDate"], errors="coerce")
    done = pd.to_datetime(df["CompletionDate"], errors="coerce")
    days = (done - created).dt.total_seconds() / 86400
    print(f"\nCreated {created.min()} -> {created.max()}")
    print(f"Completed {done.min()} -> {done.max()}")
    print("Duration computed from dates (days):", days.describe().round(1).to_dict())
    print(f"Completed BEFORE created: {(days < 0).sum():,}")
    dopen = df["DaysOpen"]
    print(f"DaysOpen present: {dopen.notna().sum():,} of {len(df):,}; "
          f"null although completed: {(dopen.isna() & done.notna()).sum():,}")
    ok = dopen.notna() & days.notna()
    print(f"DaysOpen == floor(computed days): {(dopen[ok] == np.floor(days[ok])).mean():.1%}"
          "   -> compute your own duration from the two dates")
    print("\nState:\n" + df["State"].value_counts(dropna=False).to_string())
    print("\nTop categories:\n" + df["Category"].value_counts().head(8).to_string())
    print(f"\nStudy families: {df['Study'].str[:5].nunique()}, studies: {df['Study'].nunique()}, "
          f"sites: {df['Site'].nunique()}")


def profile_imd(df):
    h(f"IM_PD: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    show_nulls(df)
    idc = find_col(df, "ID")
    print(f"\nID unique: {df[idc].is_unique}")
    pre = prefix(df[idc])
    print("ID prefixes:\n" + pre.value_counts(dropna=False).to_string())
    suf = suffix(df[idc])
    per = pd.DataFrame({"pre": pre, "suf": suf}).groupby("suf")["pre"].nunique()
    print(f"\nCase numbers that appear under BOTH prefixes: {(per > 1).sum():,} of {len(per):,}")
    print("  (if ~0, a converted case exists once, under one prefix only)")
    for c in ("Identifier", "Name"):
        col = find_col(df, c)
        if col:
            print(f"{col}: unique {df[col].nunique():,} / non-null {df[col].notna().sum():,}"
                  f"; examples {df[col].dropna().head(3).tolist()}")
    for c in ("LifecycleState", "Severity", "Category"):
        col = find_col(df, c)
        if col:
            print(f"\n{col}:\n" + df[col].value_counts(dropna=False).head(10).to_string())


def link_report(imd, task, tm):
    h("Do Task / Task Monitoring cases exist in IM_PD?")
    idc = find_col(imd, "ID")
    ids = set(imd[idc].astype(str))
    sufs = set(suffix(imd[idc]).dropna())
    for name, df in (("Task", task), ("TM", tm)):
        if df is None:
            continue
        i_ok = df["IssueObject_ID"].astype(str).isin(ids)
        p_ok = df["ProtocolDeviation_ID"].astype(str).isin(ids)
        print(f"{name}: IssueObject_ID found {i_ok.mean():.1%}; ProtocolDeviation_ID found {p_ok.mean():.1%}")
        print(f"   Issue only {(i_ok & ~p_ok).sum():,} | PD only {(~i_ok & p_ok).sum():,} | "
              f"both {(i_ok & p_ok).sum():,} | neither {(~i_ok & ~p_ok).sum():,}")
        print(f"   case number found under any prefix: {suffix(df['IssueObject_ID']).isin(sufs).mean():.1%}")


def profile_el(el, imd, task, tm):
    h(f"Event log: {el.shape[0]:,} rows x {el.shape[1]} columns")
    print("Columns:", list(el.columns))
    show_nulls(el)
    idc = find_col(el, "ID", "EVENT_ID")
    el = el.copy()
    el["_act"] = prefix(el[idc])
    el["_key"] = suffix(el[idc])
    vc = el[idc].value_counts()
    print(f"\n{idc}: {el[idc].nunique():,} distinct of {len(el):,} rows; "
          f"IDs used more than once: {(vc > 1).sum():,}; max repeats {vc.max()}")
    print(f"Exact duplicate rows (all columns identical): {el.drop(columns=['_act', '_key']).duplicated().sum():,}")
    ts = find_col(el, "TIMESTAMP", "EVENTTIME")
    if ts:
        t = pd.to_datetime(el[ts], errors="coerce")
        print(f"{ts}: {t.min()} -> {t.max()}; unparseable {t.isna().sum():,}; "
              f"midnight-only (no time of day) {(t.dt.normalize() == t).mean():.1%}")
    print("\nActivity (part of ID before the first underscore), top 60:")
    print(el["_act"].value_counts().head(60).to_string())

    if imd is None:
        print("\n(IM_PD.csv not found - skipping the join tests against it)")
        return
    h("Which event column links to a case?")
    idc_imd = find_col(imd, "ID")
    refs = {"IM_PD.ID": set(imd[idc_imd].astype(str)),
            "IM_PD.ID number": set(suffix(imd[idc_imd]).dropna())}
    for c in ("Identifier", "Name"):
        col = find_col(imd, c)
        if col:
            refs[f"IM_PD.{col}"] = set(imd[col].dropna().astype(str))
    if task is not None:
        refs["Task.TaskID"] = set(task["TaskID"].dropna().astype(str))
    if tm is not None:
        refs["TM.TaskMonitoringFollowUpID"] = set(tm["TaskMonitoringFollowUpID"].dropna().astype(str))

    rows = []
    cands = [c for c in el.columns
             if c != "_act" and (pd.api.types.is_string_dtype(el[c]) or pd.api.types.is_object_dtype(el[c]))]
    sample = el.sample(min(len(el), 200_000), random_state=0)
    for c in cands:
        vals = sample[c].dropna().astype(str)
        if vals.empty:
            continue
        for rname, ref in refs.items():
            rows.append((c, rname, rate(vals, ref)))
        if vals.str.contains("_").any():
            for rname, ref in refs.items():
                rows.append((f"{c} (part after '_')", rname, rate(suffix(vals).dropna(), ref)))
    out = pd.DataFrame(rows, columns=["event column", "matches", "share"])
    out = out[out["share"] >= 0.05].sort_values("share", ascending=False)
    print(out.head(25).to_string(index=False, formatters={"share": "{:.1%}".format}) if len(out)
          else "no event column matches any reference above 5% - paste the column list and I will look")
    print("\nRead it as: an event column + reference with a share near 100% is your join key.")
    print("'_key' is the part of the event ID after the first underscore.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--task", default="Task.csv")
    ap.add_argument("--tm", default="TM.csv")
    ap.add_argument("--imd", default="IM_PD.csv")
    ap.add_argument("--el", default="EL_IM_PD.csv")
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    sys.stdout = Tee(d / "profile_report.txt")

    paths = {k: d / getattr(a, k) for k in ("task", "tm", "imd", "el")}
    data = {}
    for k, p in paths.items():
        if p.exists():
            data[k] = load(p)
            print(f"loaded {p.name}: {len(data[k]):,} rows")
        else:
            data[k] = None
            print(f"MISSING {p.name} (skipped)")

    if data["task"] is not None:
        profile_child("Task", data["task"])
    if data["tm"] is not None:
        profile_child("TM (Task Monitoring follow-ups)", data["tm"])
    if data["imd"] is not None:
        profile_imd(data["imd"])
        link_report(data["imd"], data["task"], data["tm"])
    if data["el"] is not None:
        profile_el(data["el"], data["imd"], data["task"], data["tm"])
    print(f"\nReport saved to {d / 'profile_report.txt'}")


if __name__ == "__main__":
    main()
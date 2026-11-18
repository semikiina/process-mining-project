#!/usr/bin/env python3
"""
Step 3 - KPIs for the three main research questions + baseline effort for the value case.

Usage:
    python 03_kpis.py --dir /path/to/csv/folder        # folder that contains model/ from step 2

Reads  <dir>/model/*.csv   Writes <dir>/kpi/*.csv and kpi_report.txt

KPI blocks
    K1  Resolution time per case type                              (RQ2 / RQ5)
    K2  Waiting time between lifecycle events, long waits (>=30 d)  (RQ2 bottlenecks)
    K3  Rework: reopenings, tasks finished after case closure       (RQ3)
    K4  Benchmark by site / country / study / category / severity    (RQ5)
    K5  Open and stuck cases (no event for 90+ days)                 (RQ2)
    K6  Baseline effort of rework in hours (10 / 30 / 60 minutes)    (value case)

Notes
    * Waits use LIFECYCLE events only (rows without a task / follow-up key), because tasks run in parallel
      to the case and would distort the gap between consecutive events.
    * Resolution time = CreatedDate -> ResolvedDate, resolved cases only. For PDs, identified -> resolved is also given.
    * Everything is a pattern in the data, not proof of cause.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)
LONG = 30          # days: Novo's internal expectation (PD target 30 days)
STUCK = 90         # days without activity (definition from Novo Q&A 1)
EFFORT = {"minor": 10, "major": 30, "downgrade": 60}   # minutes, from the case material


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def h(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def sev_group(s):
    x = s.fillna("").astype(str).str.lower()
    return np.where(x.str.contains("minor"), "minor", np.where(x.str.contains("major"), "major", "other/unknown"))


def dist(x):
    x = x.dropna()
    x = x[x >= 0]
    if x.empty:
        return pd.Series(dtype=float)
    return pd.Series({"n": len(x), "median": x.median(), "mean": x.mean(), "p75": x.quantile(.75),
                      "p90": x.quantile(.90), f"share>{LONG}d": (x > LONG).mean()}).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--min-other", type=int, default=100)
    ap.add_argument("--reopen-regex", default="reopen")
    ap.add_argument("--start", choices=["identified", "created"], default="identified",
                    help="start of the end-to-end duration in K3/K4 (default: DateIdentified)")
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    m = d / "model"
    k = d / "kpi"
    k.mkdir(exist_ok=True)
    sys.stdout = Tee(k / "kpi_report.txt")

    ci = pd.read_csv(m / "cases_issue.csv", low_memory=False)
    cp = pd.read_csv(m / "cases_pd.csv", low_memory=False)
    for df in (ci, cp):
        for c in ("CreatedDate", "ResolvedDate", "DateIdentified"):
            if c in df:
                df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    cases = pd.concat([ci, cp], ignore_index=True)
    cases["SEV_GROUP"] = sev_group(cases["Severity"]) if "Severity" in cases else "other/unknown"
    # end-to-end duration used in K3 and K4 (negative values are data errors and are dropped)
    start_col = ("DAYS_IDENTIFIED_TO_RESOLVED"
                 if a.start == "identified" and "DAYS_IDENTIFIED_TO_RESOLVED" in cases else "DAYS_CREATED_TO_RESOLVED")
    cases["DAYS_E2E"] = cases[start_col].where(cases[start_col] >= 0)
    print(f"End-to-end duration for K3/K4 = {start_col}; negative values dropped: {(cases[start_col] < 0).sum():,}")
    ev = pd.read_csv(m / "events.csv", low_memory=False)
    ev["TIMESTAMP"] = pd.to_datetime(ev["TIMESTAMP"], format="mixed", errors="coerce")
    tasks = pd.read_csv(m / "tasks.csv", low_memory=False)
    for c in ("CreatedDate", "CompletionDate"):
        tasks[c] = pd.to_datetime(tasks[c], format="mixed", errors="coerce")
    print(f"cases {len(cases):,} | events {len(ev):,} | tasks {len(tasks):,}")

    # ------------------------------------------------------------ K1
    h("K1  Resolution time (days), resolved cases")
    res = cases[cases["ResolvedDate"].notna()]
    rows = {}
    for t, g in res.groupby("CASE_TYPE"):
        for lab, col in (("created -> resolved", "DAYS_CREATED_TO_RESOLVED"),
                         ("identified -> resolved", "DAYS_IDENTIFIED_TO_RESOLVED")):
            if col in g:
                s_ = dist(g[col])
                s_["dropped_negative"] = int((g[col] < 0).sum())
                rows[f"{t}: {lab}"] = s_
    k1 = pd.DataFrame(rows).T
    print(k1.to_string())
    k1.to_csv(k / "k1_resolution_time.csv")
    neg = (res["DAYS_CREATED_TO_RESOLVED"] < 0).sum()
    print(f"resolved before created (ignored): {neg:,}")

    # ------------------------------------------------------------ K2
    h(f"K2  Waiting time between lifecycle events (long wait = {LONG}+ days)")
    lc = ev[~ev["HAS_TASK"] & ~ev["HAS_TM"] & ev["CASE_IN_IMPD"]].copy()
    lc = lc.sort_values(["CASE_ID", "TIMESTAMP", "ROW_ORDER"], kind="stable")
    g = lc.groupby("CASE_ID")
    lc["PREV_ACT"] = g["ACTIVITY"].shift()
    lc["PREV_TS"] = g["TIMESTAMP"].shift()
    lc["WAIT_DAYS"] = (lc["TIMESTAMP"] - lc["PREV_TS"]).dt.total_seconds() / 86400
    w = lc.dropna(subset=["WAIT_DAYS"]).copy()
    w["IS_LONG"] = w["WAIT_DAYS"] >= LONG
    tot = w.groupby("CASE_TYPE")["WAIT_DAYS"].sum()
    print("Case-level view:")
    summ = []
    for t, gg in w.groupby("CASE_TYPE"):
        cases_w = gg.groupby("CASE_ID")["IS_LONG"].any()
        summ.append({"type": t, "cases with 2+ events": len(cases_w), "cases with a long wait": int(cases_w.sum()),
                     "share of cases": round(cases_w.mean(), 3),
                     "share of all waiting days inside long waits": round(gg.loc[gg.IS_LONG, "WAIT_DAYS"].sum() / tot[t], 3)})
    print(pd.DataFrame(summ).to_string(index=False))
    pd.DataFrame(summ).to_csv(k / "k2_case_summary.csv", index=False)
    tr = (w.groupby(["CASE_TYPE", "PREV_ACT", "ACTIVITY"])
          .agg(n=("WAIT_DAYS", "size"), median_days=("WAIT_DAYS", "median"), p75_days=("WAIT_DAYS", lambda x: x.quantile(.75)),
               total_days=("WAIT_DAYS", "sum"), n_long=("IS_LONG", "sum")).reset_index())
    tr["share_of_type_days"] = tr["total_days"] / tr["CASE_TYPE"].map(tot)
    tr = tr.sort_values(["CASE_TYPE", "total_days"], ascending=[True, False]).round(3)
    tr.to_csv(k / "k2_transitions.csv", index=False)
    for t, gg in tr.groupby("CASE_TYPE"):
        print(f"\nTop transitions by total waiting days - {t} (full table in k2_transitions.csv):")
        print(gg.head(12).drop(columns="CASE_TYPE").to_string(index=False))

    # ------------------------------------------------------------ K3
    h("K3  Rework")
    pat = re.compile(a.reopen_regex)
    ev["_act_norm"] = norm(ev["ACTIVITY"])
    reopen = ev[ev["_act_norm"].str.contains(pat) & ev["CASE_IN_IMPD"]]
    print("Activities treated as reopenings (check this list!):")
    print(reopen["ACTIVITY"].value_counts().to_string() if len(reopen) else "none found - use --reopen-regex")
    rc = reopen.groupby("CASE_ID").size().rename("N_REOPEN")
    cases = cases.merge(rc, left_on="ID", right_index=True, how="left")
    cases["N_REOPEN"] = cases["N_REOPEN"].fillna(0).astype(int)
    r = cases.groupby("CASE_TYPE").agg(cases=("ID", "size"), reopened=("N_REOPEN", lambda x: (x > 0).sum()),
                                       reopen_events=("N_REOPEN", "sum"))
    r["share_reopened"] = (r["reopened"] / r["cases"]).round(3)
    print("\n" + r.to_string())
    r.to_csv(k / "k3_reopenings.csv")
    resd = cases[cases["ResolvedDate"].notna()]
    cmp_ = resd.groupby(["CASE_TYPE", resd["N_REOPEN"] > 0])["DAYS_E2E"].median().unstack()
    cmp_.columns = ["median days: never reopened", "median days: reopened"][:len(cmp_.columns)]
    print("\nResolution time, reopened vs not:\n" + cmp_.round(1).to_string())
    cmp_.to_csv(k / "k3_reopen_vs_not.csv")
    keep = [c for c in ("ID", "CASE_TYPE", "SEV_GROUP", "Study", "StudyCountry", "StudySite", "Category", "DAYS_E2E",
                        "DAYS_CREATED_TO_RESOLVED", "N_REOPEN", "ResolvedDate") if c in cases.columns]
    cases[keep].to_csv(k / "cases_kpi.csv", index=False)   # one row per case, used by the plot script

    ct = cases.set_index("ID")["ResolvedDate"]
    tk = tasks.copy()
    tk["CASE_ID"] = np.where(tk["ISSUE_IN_IMPD"], tk["IssueObject_ID"], np.where(tk["PD_IN_IMPD"], tk["ProtocolDeviation_ID"], None))
    tk["CASE_RESOLVED"] = tk["CASE_ID"].map(ct)
    tk["AFTER_CLOSURE"] = tk["CompletionDate"] > (tk["CASE_RESOLVED"] + pd.Timedelta(days=1))
    ok = tk["CASE_RESOLVED"].notna()
    print(f"\nTasks completed more than 1 day AFTER their case was resolved: {int(tk.loc[ok, 'AFTER_CLOSURE'].sum()):,} "
          f"of {int(ok.sum()):,} tasks on resolved cases ({tk.loc[ok, 'AFTER_CLOSURE'].mean():.1%})")
    print("(Novo Q&A 2: tasks continuing after closure count as non-conformance; follow-ups do not.)")

    # ------------------------------------------------------------ K4
    h(f"K4  Benchmark (resolved cases; sites need {a.min_site}+ cases, others {a.min_other}+)")
    dims = [("StudySite", a.min_site), ("StudyCountry", a.min_other), ("Study", a.min_other),
            ("Category", a.min_other), ("SEV_GROUP", a.min_other), ("CASE_TYPE", 1)]
    parts = []
    rb = cases[cases["DAYS_E2E"].notna()]
    for col, mn in dims:
        if col not in rb.columns:
            continue
        gg = rb.groupby(col)["DAYS_E2E"]
        t = pd.DataFrame({"dimension": col, "value": gg.size().index, "n": gg.size().values,
                          "median_days": gg.median().values, "share_over_30d": gg.apply(lambda x: (x > LONG).mean()).values,
                          "excess_days_over_30": gg.apply(lambda x: (x - LONG).clip(lower=0).sum()).values})
        parts.append(t[t["n"] >= mn])
    bench = pd.concat(parts, ignore_index=True).round(2)
    bench.to_csv(k / "k4_benchmark.csv", index=False)
    for col, _ in dims:
        sub = bench[bench["dimension"] == col].sort_values("excess_days_over_30", ascending=False)
        if len(sub):
            print(f"\n{col}: {len(sub)} groups - biggest contributors to days beyond {LONG}:")
            print(sub.head(8).drop(columns="dimension").to_string(index=False))
    tot_exc = (rb["DAYS_E2E"] - LONG).clip(lower=0).sum()
    print(f"\nTotal case-days beyond {LONG} days across all resolved cases: {tot_exc:,.0f}")

    # ------------------------------------------------------------ K5
    h(f"K5  Open cases and stuck cases (no event for {STUCK}+ days)")
    ref = ev["TIMESTAMP"].max()
    last = ev[ev["CASE_IN_IMPD"]].groupby("CASE_ID")["TIMESTAMP"].max()
    op = cases[cases["ResolvedDate"].isna()].copy()
    op["LAST_EVENT"] = op["ID"].map(last)
    op["IDLE_DAYS"] = (ref - op["LAST_EVENT"]).dt.total_seconds() / 86400
    op["STUCK"] = op["IDLE_DAYS"] > STUCK
    print(f"reference date (latest event in data): {ref}")
    s = op.groupby("CASE_TYPE").agg(open_cases=("ID", "size"), stuck=("STUCK", "sum"))
    s["share_stuck"] = (s["stuck"] / s["open_cases"]).round(3)
    print(s.to_string())
    if "LifecycleState" in op:
        print("\nstuck cases by lifecycle state:\n" + op[op.STUCK]["LifecycleState"].value_counts(dropna=False).to_string())
    s.to_csv(k / "k5_stuck.csv")
    op[["ID", "CASE_TYPE", "LifecycleState", "LAST_EVENT", "IDLE_DAYS", "STUCK"]
       if "LifecycleState" in op else ["ID", "CASE_TYPE", "LAST_EVENT", "IDLE_DAYS", "STUCK"]
       ].to_csv(k / "k5_open_cases.csv", index=False)

    # ------------------------------------------------------------ K6
    h("K6  Baseline effort of rework (NOT savings - the starting point for the value case)")
    rp = reopen.merge(cases[["ID", "SEV_GROUP"]], left_on="CASE_ID", right_on="ID", how="left")
    base = rp.groupby(["CASE_TYPE", "SEV_GROUP"]).size().rename("reopen_events").reset_index()
    base["minutes_each"] = base["SEV_GROUP"].map(EFFORT)
    base["hours"] = (base["reopen_events"] * base["minutes_each"].fillna(0) / 60).round(1)
    print(base.to_string(index=False))
    print("(minutes_each is empty for 'other/unknown' severity - decide a rule and document it)")
    # downgrade = any Protocol Deviation -> Issue conversion (both name variants)
    dg = ev[ev["IS_CONVERSION"] & norm(ev["ACTIVITY"]).str.contains("protocoldeviation.*toissue", regex=True)]
    print(f"\nDowngrade events (PD -> Issue): {len(dg):,} x {EFFORT['downgrade']} min = "
          f"{len(dg) * EFFORT['downgrade'] / 60:,.0f} hours")
    print(dg["ACTIVITY"].value_counts().to_string())
    base.to_csv(k / "k6_rework_effort.csv", index=False)

    print(f"\nDone. Files in {k}")


if __name__ == "__main__":
    main()

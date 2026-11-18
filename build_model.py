#!/usr/bin/env python3
"""
Step 2 - build the object-centric model from the four CSVs.

Usage:
    python 02_build_model.py --dir /path/to/csv/folder            # CSV outputs
    python 02_build_model.py --dir /path/to/csv/folder --ocel     # + OCEL 2.0 file (needs pm4py)

Outputs go to <dir>/model/ :
    cases_issue.csv, cases_pd.csv      one row per Issue / PD (all IM_PD columns + computed durations)
    tasks.csv, tms.csv                 one row per task / follow-up (+ link flags, computed duration)
    events.csv                         every event with a UNIQUE EVENT_KEY and its links
    log_issue.csv, log_pd.csv          flat event logs (case = Issue / PD) for Disco, ProM, PM4Py
    log_task.csv, log_tm.csv           flat event logs (case = task / follow-up)
    model_ocel.sqlite                  only with --ocel
    build_report.txt                   counts and checks

Design decisions (all from the data profile):
    * Event -> case:  'Lead Object Id' equals IM_PD.ID for 100% of events, its prefix gives the type.
    * Event -> task / follow-up:  the part after '_' in 'Task Id' / 'TaskMonitoringFollowUp Id'.
    * Event ID:  own row-number key (the source 'Id' repeats, but no rows are identical, so nothing is dropped).
    * Task / TM -> case:  both IssueObject_ID and ProtocolDeviation_ID are always filled with the same
      number, so we check which of the two really exists in IM_PD. Rows with neither are KEPT, flagged.
    * Durations are computed from dates (DaysOpen in the source is unreliable).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def pre(s):
    return s.astype("string").str.split("_", n=1).str[0]


def suf(s):
    return s.astype("string").str.split("_", n=1).str[1]


def key_of(s):
    """'Task_ABC123' -> 'ABC123'; a value without underscore is kept as is."""
    return s.astype("string").str.split("_", n=1).str[-1]


def dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")


def days(a, b):
    return (b - a).dt.total_seconds() / 86400


def norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="Task.csv")
    ap.add_argument("--tm", default="TM.csv")
    ap.add_argument("--imd", default="IM_PD.csv")
    ap.add_argument("--el", default="EL_IM_PD.csv")
    ap.add_argument("--ocel", action="store_true", help="also write an OCEL 2.0 sqlite file (needs pm4py)")
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    out = Path(a.out).expanduser() if a.out else d / "model"
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "build_report.txt")

    imd = pd.read_csv(d / a.imd, low_memory=False)
    el = pd.read_csv(d / a.el, low_memory=False)
    task = pd.read_csv(d / a.task, low_memory=False)
    tm = pd.read_csv(d / a.tm, low_memory=False)
    print(f"loaded: IM_PD {len(imd):,} | events {len(el):,} | tasks {len(task):,} | follow-ups {len(tm):,}")

    # ---------------- cases ----------------
    dates(imd, ["CreatedDate", "DateIdentified", "DateOfViolationDeviation", "CorrectionDate", "ImplementedDate",
                "LastModifiedDate", "ResolvedDate", "ReviewStartDate", "ReviewCompletedDate", "ReviewDueDate"])
    imd["CASE_TYPE"] = pre(imd["ID"])
    imd["CASE_NO"] = suf(imd["ID"])
    unknown = ~imd["CASE_TYPE"].isin(["Issue", "ProtocolDeviation"])
    print(f"IM_PD rows with an unknown ID prefix: {unknown.sum():,}")
    imd["IS_CONVERTED_PAIR"] = imd.groupby("CASE_NO")["ID"].transform("nunique") > 1
    imd["DAYS_CREATED_TO_RESOLVED"] = days(imd["CreatedDate"], imd["ResolvedDate"])
    if "DateIdentified" in imd:
        imd["DAYS_IDENTIFIED_TO_RESOLVED"] = days(imd["DateIdentified"], imd["ResolvedDate"])
    imd["IS_OPEN"] = imd["ResolvedDate"].isna()
    case_ids = set(imd["ID"])
    imd[imd["CASE_TYPE"] == "Issue"].to_csv(out / "cases_issue.csv", index=False)
    imd[imd["CASE_TYPE"] == "ProtocolDeviation"].to_csv(out / "cases_pd.csv", index=False)
    print(f"Issues {(imd.CASE_TYPE == 'Issue').sum():,} | PDs {(imd.CASE_TYPE == 'ProtocolDeviation').sum():,} | "
          f"rows that belong to a converted pair {imd.IS_CONVERTED_PAIR.sum():,}")

    # ---------------- tasks and follow-ups ----------------
    for name, df, keycol in (("tasks", task, "TaskID"), ("tms", tm, "TaskMonitoringFollowUpID")):
        dates(df, ["CreatedDate", "CompletionDate"])
        df["CASE_NO"] = suf(df["IssueObject_ID"])
        df["ISSUE_IN_IMPD"] = df["IssueObject_ID"].isin(case_ids)
        df["PD_IN_IMPD"] = df["ProtocolDeviation_ID"].isin(case_ids)
        df["IN_IMPD"] = df["ISSUE_IN_IMPD"] | df["PD_IN_IMPD"]
        df["DAYS_CREATED_TO_COMPLETED"] = days(df["CreatedDate"], df["CompletionDate"])
        df["IS_OPEN"] = df["CompletionDate"].isna()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: matched to an Issue only {(df.ISSUE_IN_IMPD & ~df.PD_IN_IMPD).sum():,} | PD only "
              f"{(~df.ISSUE_IN_IMPD & df.PD_IN_IMPD).sum():,} | both {(df.ISSUE_IN_IMPD & df.PD_IN_IMPD).sum():,} | "
              f"neither (kept) {(~df.IN_IMPD).sum():,}")

    # ---------------- events ----------------
    act_col = "Activity_prettified" if "Activity_prettified" in el.columns else "Activity"
    print(f"activity label used: {act_col}")
    el = el.reset_index(drop=True)
    ev = pd.DataFrame({
        "EVENT_KEY": "EV_" + pd.Series(np.arange(1, len(el) + 1)).astype(str),
        "ROW_ORDER": np.arange(1, len(el) + 1),
        "SOURCE_ID": el["Id"],
        "ACTIVITY": el[act_col],
        "ACTIVITY_RAW": el["Activity"],
        "ATTRIBUTE": el.get("Attribute"),
        "OLD_VALUE": el.get("OldValue"),
        "NEW_VALUE": el.get("NewValue"),
        "TIMESTAMP": pd.to_datetime(el["Timestamp"], format="mixed", errors="coerce"),
        "USER": el.get("UserName"),
        "EXECUTION_TYPE": el.get("ExecutionType"),
        "CASE_ID": el["Lead Object Id"],
        "CASE_TYPE": pre(el["Lead Object Id"]),
        "CASE_NAME": el.get("Name"),
        "TASK_KEY": key_of(el["Task Id"]) if "Task Id" in el else pd.NA,
        "TM_KEY": key_of(el["TaskMonitoringFollowUp Id"]) if "TaskMonitoringFollowUp Id" in el else pd.NA,
    })
    ev["CASE_IN_IMPD"] = ev["CASE_ID"].isin(case_ids)
    ev["TASK_IN_TASKS"] = ev["TASK_KEY"].isin(set(task["TaskID"].astype(str)))
    ev["TM_IN_TMS"] = ev["TM_KEY"].isin(set(tm["TaskMonitoringFollowUpID"].astype(str)))
    ev["HAS_TASK"] = ev["TASK_KEY"].notna()
    ev["HAS_TM"] = ev["TM_KEY"].notna()
    conv = norm(ev["ACTIVITY"]) + "|" + norm(ev["ACTIVITY_RAW"])
    ev["IS_CONVERSION"] = conv.str.contains(r"issue.*to.*protocoldeviation|protocoldeviation.*to.*issue", regex=True)
    pair_ids = set(imd.loc[imd["IS_CONVERTED_PAIR"], "ID"])
    ev["CASE_IS_CONVERTED_PAIR"] = ev["CASE_ID"].isin(pair_ids)
    ev = ev.sort_values(["CASE_ID", "TIMESTAMP", "ROW_ORDER"], kind="stable")
    ev.to_csv(out / "events.csv", index=False)

    print("\n--- event checks ---")
    print(f"events {len(ev):,} | unique EVENT_KEY {ev.EVENT_KEY.nunique():,} | missing timestamps {ev.TIMESTAMP.isna().sum():,}")
    print(f"events whose case is NOT in IM_PD: {(~ev.CASE_IN_IMPD).sum():,}")
    print(f"events with a Task key {ev.HAS_TASK.sum():,} (found in Task.csv: {ev.TASK_IN_TASKS.sum():,})")
    print(f"events with a follow-up key {ev.HAS_TM.sum():,} (found in TM.csv: {ev.TM_IN_TMS.sum():,})")
    print(f"conversion events (Issue<->PD): {ev.IS_CONVERSION.sum():,}")
    print(ev.loc[ev.IS_CONVERSION, "ACTIVITY"].value_counts().to_string())
    print("check these names - if empty or wrong, the conversion pattern needs adjusting")
    print("\nevents per case type:\n" + ev.CASE_TYPE.value_counts().to_string())

    # ---------------- flat logs (one per object type) ----------------
    def flat(df, case_col, path):
        f = df.rename(columns={case_col: "case:concept:name", "ACTIVITY": "concept:name",
                               "TIMESTAMP": "time:timestamp", "USER": "org:resource"})
        cols = ["case:concept:name", "concept:name", "time:timestamp", "org:resource", "EVENT_KEY",
                "OLD_VALUE", "NEW_VALUE", "EXECUTION_TYPE", "HAS_TASK", "HAS_TM"]
        f = f.dropna(subset=["time:timestamp"])[[c for c in cols if c in f.columns]]
        f.to_csv(path, index=False)
        return f["case:concept:name"].nunique(), len(f)

    print("\n--- flat logs ---")
    for label, sub, col, fn in (
        ("Issue", ev[ev.CASE_TYPE == "Issue"], "CASE_ID", "log_issue.csv"),
        ("PD", ev[ev.CASE_TYPE == "ProtocolDeviation"], "CASE_ID", "log_pd.csv"),
        ("Task", ev[ev.HAS_TASK], "TASK_KEY", "log_task.csv"),
        ("TM", ev[ev.HAS_TM], "TM_KEY", "log_tm.csv"),
    ):
        n_cases, n_ev = flat(sub, col, out / fn)
        print(f"{fn}: {n_cases:,} cases, {n_ev:,} events")
    print("note: log_issue / log_pd contain task events too (they carry the case id). "
          "Filter HAS_TASK == False for the pure case lifecycle.")

    # ---------------- optional OCEL 2.0 ----------------
    if a.ocel:
        build_ocel(out, imd, task, tm, ev)

    print(f"\nDone. Files are in {out}")


def build_ocel(out, imd, task, tm, ev):
    print("\n--- OCEL 2.0 ---")
    try:
        import pm4py
        from pm4py.objects.ocel.obj import OCEL
    except Exception as e:  # noqa
        print(f"pm4py not available ({e}); skipped")
        return
    e = ev.dropna(subset=["TIMESTAMP"])
    events = pd.DataFrame({"ocel:eid": e["EVENT_KEY"], "ocel:timestamp": e["TIMESTAMP"],
                           "ocel:activity": e["ACTIVITY"].astype(str)})
    objects = pd.concat([
        pd.DataFrame({"ocel:oid": imd["ID"], "ocel:type": imd["CASE_TYPE"]}),
        pd.DataFrame({"ocel:oid": task["ID"], "ocel:type": "Task"}),
        pd.DataFrame({"ocel:oid": tm["ID"], "ocel:type": "TaskMonitoring"}),
    ], ignore_index=True).drop_duplicates("ocel:oid")
    task_map = dict(zip(task["TaskID"].astype(str), task["ID"]))
    tm_map = dict(zip(tm["TaskMonitoringFollowUpID"].astype(str), tm["ID"]))
    pieces = []
    base = e[e["CASE_IN_IMPD"]]
    pieces.append(pd.DataFrame({"ocel:eid": base["EVENT_KEY"], "ocel:oid": base["CASE_ID"], "ocel:qualifier": "case"}))
    # conversion events also touch the other-type row of the same case number
    imd_idx = imd.set_index("ID")["CASE_NO"]
    by_no = imd.groupby("CASE_NO")["ID"].apply(list).to_dict()
    cv = base[base["IS_CONVERSION"] & base["CASE_IS_CONVERTED_PAIR"]]
    if len(cv):
        rows = []
        for eid, cid in zip(cv["EVENT_KEY"], cv["CASE_ID"]):
            for other in by_no.get(imd_idx.get(cid), []):
                if other != cid:
                    rows.append((eid, other, "conversion"))
        pieces.append(pd.DataFrame(rows, columns=["ocel:eid", "ocel:oid", "ocel:qualifier"]))
    t = e[e["HAS_TASK"] & e["TASK_IN_TASKS"]]
    pieces.append(pd.DataFrame({"ocel:eid": t["EVENT_KEY"], "ocel:oid": t["TASK_KEY"].astype(str).map(task_map),
                                "ocel:qualifier": "task"}))
    m = e[e["HAS_TM"] & e["TM_IN_TMS"]]
    pieces.append(pd.DataFrame({"ocel:eid": m["EVENT_KEY"], "ocel:oid": m["TM_KEY"].astype(str).map(tm_map),
                                "ocel:qualifier": "followup"}))
    rel = pd.concat(pieces, ignore_index=True).dropna()
    rel = rel.merge(objects.rename(columns={"ocel:type": "ocel:type"}), on="ocel:oid", how="inner")
    rel = rel.merge(events[["ocel:eid", "ocel:activity", "ocel:timestamp"]], on="ocel:eid", how="inner")
    events = events[events["ocel:eid"].isin(rel["ocel:eid"])]
    ocel = OCEL(events=events, objects=objects, relations=rel)
    path = str(out / "model_ocel.sqlite")
    pm4py.write_ocel2_sqlite(ocel, path)
    print(f"events {len(events):,} | objects {len(objects):,} | relations {len(rel):,}")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
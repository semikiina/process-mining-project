#!/usr/bin/env python3
"""
Step 4 - figures for the report and poster.

Usage:
    python 04_plots.py --dir /path/to/csv/folder      # folder that contains kpi/ from step 3 (rerun 03 first)

Reads  <dir>/kpi/*.csv    Writes <dir>/kpi/plots/*.png (200 dpi, for the report) and *.pdf (vector, for the poster)

Figures
    fig1_resolution_time     cumulative share of cases resolved by day, Issue vs PD, 30-day line
    fig2_long_waits          share of cases with a 30+ day wait, and share of waiting days inside such waits
    fig3_bottlenecks         transitions that hold the most waiting days (per case type)
    fig4_reopenings          resolution time with and without reopening
    fig5_studies             studies that hold the most days beyond 30
    fig6_categories          resolution time by category
    fig7_stuck_cases         open cases by days since last activity

Colour rules: Issue is always blue and Protocol Deviation always orange; one-hue blue for magnitude charts.
"""
import argparse
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TYPE_COLOR = {"Issue": BLUE, "ProtocolDeviation": ORANGE}
TYPE_NAME = {"Issue": "Issue", "ProtocolDeviation": "Protocol deviation"}
LONG = 30

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1, "font.size": 10, "legend.frameon": False,
})


def tidy(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def heading(fig, title, sub=None, top=0.80):
    fig.text(0.012, 0.965, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if sub:
        fig.text(0.012, 0.905, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=top)


def save(fig, out, name):
    # bbox_inches="tight" widens the canvas if any label sticks out, so nothing is cropped
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", name)


def clean_cat(s):
    s = re.sub(r"__(c|ctms)$", "", str(s))
    return s.replace("_", " ").replace("andor", "and/or")


def thousands(x, _):
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------------------
def fig1(cases, out):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    over = {}
    for t in ("Issue", "ProtocolDeviation"):
        x = np.sort(cases.loc[(cases.CASE_TYPE == t) & cases.DAYS_E2E.notna(), "DAYS_E2E"].to_numpy())
        if not len(x):
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=TYPE_COLOR[t], lw=2, label=f"{TYPE_NAME[t]} (n={len(x):,})")
        within = (x <= LONG).mean()
        over[t] = 1 - within
        ax.plot([LONG], [within], "o", ms=7, color=TYPE_COLOR[t], mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{TYPE_NAME[t]}: {within:.0%} within 30 d", (LONG, within),
                    xytext=(10, -14 if t == "Issue" else 6), textcoords="offset points", fontsize=9, color=INK2)
    ax.axvline(LONG, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Days from identification to resolution (axis cut at 365 days)")
    ax.set_ylabel("Share of resolved cases")
    ax.legend(loc="lower right")
    tidy(ax)
    if "Issue" in over and "ProtocolDeviation" in over:
        t = f"{over['Issue']:.0%} of Issues and {over['ProtocolDeviation']:.0%} of PDs take longer than 30 days"
    else:
        t = "Resolution time"
    heading(fig, t, "Cumulative share of resolved cases. 30 days is the target for protocol deviations.")
    save(fig, out, "fig1_resolution_time")


def fig2(k2s, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    measures = [("share of cases", "Cases with at least one\nwait of 30+ days"),
                ("share of all waiting days inside long waits", "Waiting days that sit\ninside 30+ day waits")]
    types = [t for t in ("Issue", "ProtocolDeviation") if t in set(k2s["type"])]
    w = 0.34
    for i, t in enumerate(types):
        row = k2s[k2s["type"] == t].iloc[0]
        xs = np.arange(len(measures)) + (i - (len(types) - 1) / 2) * (w + 0.03)
        vals = [row[m] for m, _ in measures]
        ax.bar(xs, vals, width=w, color=TYPE_COLOR[t], label=TYPE_NAME[t], edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels([lab for _, lab in measures], color=INK2)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper right", ncol=2)
    tidy(ax)
    heading(fig, "Most waiting time is spent in a few very long waits",
            "Waits are measured between consecutive lifecycle events of the same case.")
    save(fig, out, "fig2_long_waits")


def fig3(tr, out, top=7):
    types = [t for t in ("Issue", "ProtocolDeviation") if t in set(tr["CASE_TYPE"])]
    fig, axes = plt.subplots(len(types), 1, figsize=(7.6, 4.0 * len(types)), squeeze=False)
    for ax, t in zip(axes[:, 0], types):
        d = tr[(tr.CASE_TYPE == t) & (tr.PREV_ACT != tr.ACTIVITY)].nlargest(top, "total_days").iloc[::-1]
        labels = [textwrap.fill(f"{p} → {a}", 46) for p, a in zip(d.PREV_ACT, d.ACTIVITY)]
        ax.barh(labels, d.share_of_type_days, height=0.55, color=TYPE_COLOR[t], edgecolor=SURFACE)
        for y, (s, m) in enumerate(zip(d.share_of_type_days, d.median_days)):
            pct = f"{s:.1%}" if s < 0.01 else f"{s:.0%}"
            ax.text(s + 0.004, y, f"{pct}  (median {m:.0f} d)", va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(0, max(d.share_of_type_days.max() * 1.45, 0.05))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(TYPE_NAME[t], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=INK2, labelsize=8.5)
        tidy(ax, "x")
    axes[-1, 0].set_xlabel("Share of the case type's total waiting days")
    fig.subplots_adjust(left=0.46, right=0.97, hspace=0.35, bottom=0.07 if len(types) > 1 else 0.2)
    heading(fig, "Where the waiting days end up",
            "Top transitions by total waiting days. Repeated 'other changes' edits are left out.",
            top=0.83 if len(types) > 1 else 0.78)
    save(fig, out, "fig3_bottlenecks")


def fig4(cmp_, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    c_never = [c for c in cmp_.columns if "never" in c][0]
    c_re = [c for c in cmp_.columns if c != c_never][0]
    types = [t for t in ("Issue", "ProtocolDeviation") if t in cmp_.index]
    w = 0.34
    for j, t in enumerate(types):
        for i, (col, solid) in enumerate(((c_never, False), (c_re, True))):
            x = j + (i - 0.5) * (w + 0.03)
            v = cmp_.loc[t, col]
            ax.bar(x, v, width=w, color=TYPE_COLOR[t], alpha=1 if solid else 0.42, edgecolor=SURFACE, linewidth=2)
            ax.text(x, v + 1.2, f"{v:.0f} d", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([TYPE_NAME[t] for t in types], color=INK2)
    ax.set_ylabel("Median days, identification to resolution")
    ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.42, label="Never reopened"),
                       Patch(facecolor=INK2, label="Reopened at least once")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    ax.set_ylim(0, cmp_[[c_never, c_re]].to_numpy().max() * 1.15)
    tidy(ax)
    longer = all(cmp_.loc[t, c_re] > cmp_.loc[t, c_never] for t in types)
    heading(fig, "Reopened cases take longer overall" if longer else "Resolution time with and without reopening",
            "Median days from identification to resolution. Reopened cases are longer by construction.",
            top=0.82)
    fig.subplots_adjust(bottom=0.18)
    save(fig, out, "fig4_reopenings")


def fig5(bench, cases, out, top=10):
    d = bench[bench.dimension == "Study"].nlargest(top, "excess_days_over_30").iloc[::-1]
    tot = (cases["DAYS_E2E"] - LONG).clip(lower=0).sum()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig.subplots_adjust(left=0.17, right=0.97)
    ax.barh(d["value"], d["excess_days_over_30"], height=0.55, color=BLUE, edgecolor=SURFACE)
    for y, (v, n, m) in enumerate(zip(d.excess_days_over_30, d.n, d.median_days)):
        ax.text(v + tot * 0.002, y, f"{v / tot:.1%}  (n={n:,}, median {m:.0f} d)", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, d.excess_days_over_30.max() * 1.55)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel(f"Case-days beyond {LONG} days")
    tidy(ax, "x")
    top3 = d.nlargest(3, "excess_days_over_30").excess_days_over_30.sum() / tot
    heading(fig, f"The 3 largest studies hold {top3:.0%} of all days beyond 30",
            f"Top {top} studies. Share of all case-days beyond {LONG} days across resolved cases.")
    save(fig, out, "fig5_studies")


def fig6(bench, out, top=12):
    d = bench[bench.dimension == "Category"].nlargest(top, "n").sort_values("median_days")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    names = [textwrap.fill(clean_cat(v), 30) for v in d["value"]]
    fig.subplots_adjust(left=0.34, right=0.97)
    ax.barh(names, d.median_days, height=0.55, color=BLUE, edgecolor=SURFACE)
    xt = max(d.median_days.max() * 1.04, LONG * 1.06)       # labels sit in one column right of the bars
    for y, (m, s, n) in enumerate(zip(d.median_days, d.share_over_30d, d.n)):
        ax.text(xt, y, f"{m:.0f} d  ·  {s:.0%} over 30 d  ·  n={n:,}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(LONG, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, xt * 1.8)
    ax.set_xlabel("Median days, identification to resolution")
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=8.5)
    tidy(ax, "x")
    slow = d.iloc[-1]
    heading(fig, f"{clean_cat(slow['value']).capitalize()} cases are the slowest category",
            f"The {top} largest categories, by median time. Dashed line: 30 days.")
    save(fig, out, "fig6_categories")


def fig7(op, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cap = 730
    bins = np.arange(0, cap + 30, 30)
    types = [t for t in ("Issue", "ProtocolDeviation") if t in set(op.CASE_TYPE)]
    data = [op.loc[op.CASE_TYPE == t, "IDLE_DAYS"].clip(upper=cap).dropna() for t in types]
    ax.hist(data, bins=bins, stacked=True, color=[TYPE_COLOR[t] for t in types],
            edgecolor=SURFACE, linewidth=1.5, label=[TYPE_NAME[t] for t in types])
    ax.axvline(90, color=INK2, lw=1, ls=(0, (4, 3)))
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)
    ax.text(95, ymax * 1.06, "90 days: counted as stuck", fontsize=9, color=INK2)
    ax.set_xlabel(f"Days since last event (values above {cap} grouped in the last bar)")
    ax.set_ylabel("Open cases")
    ax.legend(loc="upper right")
    tidy(ax)
    n, stuck = len(op), int((op.IDLE_DAYS > 90).sum())
    heading(fig, f"{stuck:,} of {n:,} open cases have had no activity for over 90 days",
            "Open cases (no resolved date), by time since their last recorded event.")
    save(fig, out, "fig7_stuck_cases")


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    k = Path(a.dir).expanduser() / "kpi"
    out = k / "plots"
    out.mkdir(parents=True, exist_ok=True)

    def rd(name, **kw):
        p = k / name
        if not p.exists():
            print(f"missing {name} - rerun 03_kpis.py (skipping its figures)")
            return None
        return pd.read_csv(p, **kw)

    cases, k2s, tr = rd("cases_kpi.csv", low_memory=False), rd("k2_case_summary.csv"), rd("k2_transitions.csv")
    cmp_, bench, op = rd("k3_reopen_vs_not.csv", index_col=0), rd("k4_benchmark.csv"), rd("k5_open_cases.csv")

    jobs = [("fig1", cases is not None, lambda: fig1(cases, out)),
            ("fig2", k2s is not None, lambda: fig2(k2s, out)),
            ("fig3", tr is not None, lambda: fig3(tr, out)),
            ("fig4", cmp_ is not None, lambda: fig4(cmp_, out)),
            ("fig5", bench is not None and cases is not None, lambda: fig5(bench, cases, out)),
            ("fig6", bench is not None, lambda: fig6(bench, out)),
            ("fig7", op is not None, lambda: fig7(op, out))]
    for name, ok, fn in jobs:
        if not ok:
            continue
        try:
            fn()
        except Exception as e:  # keep going so one bad figure does not block the rest
            print(f"{name} FAILED: {type(e).__name__}: {e}")
    print(f"\nFigures are in {out}")


if __name__ == "__main__":
    main()
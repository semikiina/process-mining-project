#!/usr/bin/env python3
"""
Step 5 - Research question 5: how do sites and countries differ?

Usage (folder that holds model/ and kpi/ from steps 2-3; rerun 03 first if cases_kpi.csv is missing):
    python 05_sites_countries.py --dir /path/to/csv/folder
Options:
    --type both|Issue|ProtocolDeviation   restrict to one case type (outputs go to rq5_<type>/)
    --min-site 30 --min-country 100 --min-study 100     smallest groups that are compared
    --pool-country auto|yes|no            also pool the C-number across studies (auto = only if numbering looks global)

Reads  kpi/cases_kpi.csv  (+ model/cases_pd.csv, tasks.csv, tms.csv when present)
Writes rq5/*.csv, rq5/plots/*.png|pdf, rq5/rq5_report.txt

Method in short (so it can be explained in the report)
  * "Apples to apples": every case gets an EXPECTED duration from comparable cases (same type x severity x category;
    falls back to coarser groups when a group has < 30 cases). A site's effect = mean of log(1+days) minus expected.
    ratio 1.4 = the site's cases take about 40 % longer than comparable cases elsewhere.
  * Two views: 'mix' (adjusted for case mix only) and 'within study' (also removes the study average, so a site is
    compared with other sites of the SAME study - protocol complexity and therapeutic area cannot explain it).
  * Uncertainty: 95 % interval, and the Benjamini-Hochberg false-discovery rate across all sites tested (q < 0.05).
  * Shrinkage: DerSimonian-Laird estimate of the true spread between sites; noisy small sites are pulled to 1.
  * Everything is a pattern in the data, not proof of cause.
"""
import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 40)
LONG = 30
FDR = 0.05
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


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
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


# ------------------------------------------------------------------ statistics helpers
def pval(z):
    return np.array([math.erfc(abs(v) / math.sqrt(2)) if v == v else np.nan for v in np.asarray(z, float)])


def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if not len(idx):
        return q
    order = idx[np.argsort(p[idx])]
    r = p[order] * len(idx) / np.arange(1, len(idx) + 1)
    q[order] = np.minimum(np.minimum.accumulate(r[::-1])[::-1], 1)
    return q


def hier_expected(df, y, levels, min_n=30):
    """Expected y per row: mean of y in the finest group (levels go fine -> coarse) that has >= min_n rows."""
    exp = pd.Series(np.nan, index=df.index)
    for lv in levels:
        if lv:
            g = df.groupby(lv, dropna=False)[y]
            m, n = g.transform("mean"), g.transform("size")
        else:
            m, n = pd.Series(df[y].mean(), index=df.index), pd.Series(len(df), index=df.index)
        take = exp.isna() & (n >= min_n)
        exp[take] = m[take]
    return exp.fillna(df[y].mean())


def flags(q, direction):
    return np.where((q < FDR) & (direction > 0), "slower", np.where((q < FDR) & (direction < 0), "faster", ""))


def group_effects(df, group, resid="RESID", min_n=30):
    """Continuous outcome: per group mean residual (log scale), CI, FDR, shrinkage. Returns (table, tau)."""
    g = df.groupby(group)[resid].agg(n="size", m="mean")
    g = g[g["n"] >= min_n].copy()
    if g.empty:
        return g, float("nan")
    s = float(df[resid].std(ddof=1))
    g["se"] = s / np.sqrt(g["n"])
    g["z"] = g["m"] / g["se"]
    g["p"] = pval(g["z"].values)
    g["q"] = bh(g["p"].values)
    g["ratio"] = np.exp(g["m"])
    g["ci_lo"] = np.exp(g["m"] - 1.96 * g["se"])
    g["ci_hi"] = np.exp(g["m"] + 1.96 * g["se"])
    w = 1 / g["se"] ** 2
    mbar = float((w * g["m"]).sum() / w.sum())
    k = len(g)
    q_stat = float((w * (g["m"] - mbar) ** 2).sum())
    tau2 = max(0.0, (q_stat - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    g["m_shrunk"] = mbar + tau2 / (tau2 + g["se"] ** 2) * (g["m"] - mbar)
    g["ratio_shrunk"] = np.exp(g["m_shrunk"])
    g["flag"] = flags(g["q"].values, g["m"].values)
    return g, math.sqrt(tau2)


def group_binary(df, group, y, p, min_n=30):
    """Binary outcome: observed vs expected events per group (indirect standardisation)."""
    d = pd.DataFrame({"g": df[group], "y": df[y].astype(float), "p": df[p], "v": df[p] * (1 - df[p])})
    g = d.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"), var=("v", "sum"))
    g = g[(g["n"] >= min_n) & (g["exp"] > 0)].copy()
    if g.empty:
        return g
    g["rate"] = g["obs"] / g["n"]
    g["ratio"] = g["obs"] / g["exp"]
    sd = np.sqrt(g["var"].clip(lower=1e-9))
    g["z"] = (g["obs"] - g["exp"]) / sd
    g["p_val"] = pval(g["z"].values)
    g["q"] = bh(g["p_val"].values)
    g["ci_lo"] = ((g["obs"] - 1.96 * sd) / g["exp"]).clip(lower=0)
    g["ci_hi"] = (g["obs"] + 1.96 * sd) / g["exp"]
    g["flag"] = flags(g["q"].values, g["z"].values)
    g.index.name = group
    return g


def adj_r2(y, labels):
    lab = labels.fillna("(missing)").astype(str)
    grp = y.groupby(lab)
    n, k = len(y), grp.ngroups
    sst = float(((y - y.mean()) ** 2).sum())
    ssb = float((grp.size() * (grp.mean() - y.mean()) ** 2).sum())
    r2 = ssb / sst if sst > 0 else 0.0
    return max(0.0, 1 - (1 - r2) * (n - 1) / max(n - k, 1)), k


def spearman(a, b):
    d = pd.concat([a, b], axis=1, join="inner").dropna()
    return (float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")), len(d)) if len(d) >= 8 else (float("nan"), len(d))


def key_parts(s):
    """'P0004-T0001-C0014-S0001' -> study, country(in study), site"""
    p = s.str.split("-")
    return p.str[:2].str.join("-"), p.str[:3].str.join("-")


def show(tbl, cols, n=8, sort="ratio", asc=False, fmt=None):
    if tbl is None or tbl.empty:
        print("(nothing to show - no group has enough cases)")
        return
    t = tbl.sort_values(sort, ascending=asc).head(n)[cols].copy()
    print(t.to_string(float_format=lambda v: f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"))


# ------------------------------------------------------------------ figures
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
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", name)


def ratio_axis(ax, axis="y"):
    ticks = [0.25, 0.5, 1, 2, 4]
    fn = FuncFormatter(lambda v, _: f"×{v:g}")
    if axis == "y":
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(fn)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    else:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(fn)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def fig_funnel(sites, out, label):
    n_all = len(sites)
    s = float((sites["se"] * np.sqrt(sites["n"])).iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    xs = np.logspace(np.log10(sites["n"].min()), np.log10(sites["n"].max()), 100)
    for z in (3, -3):
        ax.plot(xs, np.exp(z * s / np.sqrt(xs)), color=INK2, lw=1, ls=(0, (4, 3)))
    ax.axhline(1, color=AXIS, lw=1)
    for fl, col, sz in (("", MUTED, 14), ("slower", BLUE, 26), ("faster", AQUA, 26)):
        d = sites[sites["flag"] == fl]
        ax.scatter(d["n"], d["ratio"], s=sz, color=col, alpha=0.5 if fl == "" else 0.95,
                   edgecolors=SURFACE if fl else "none", linewidths=0.6, zorder=3 if fl else 2)
    ax.set_xscale("log")
    ratio_axis(ax)
    lo_n, hi_n = sites["n"].min(), sites["n"].max()
    ticks = [t for t in (10, 20, 30, 50, 100, 200, 500, 1000, 2000) if lo_n * 0.9 <= t <= hi_n * 1.1]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Cases at the site (log scale)")
    ax.set_ylabel("Time vs comparable cases elsewhere")
    ax.set_ylim(min(0.2, sites["ratio"].min() * 0.9), max(5, sites["ratio"].max() * 1.1))
    mid = np.sqrt(lo_n * hi_n)
    for _, r in pd.concat([sites[sites.flag == "slower"].nlargest(1, "z"),
                           sites[sites.flag == "faster"].nsmallest(1, "z")]).iterrows():
        right = r["n"] > mid                      # labels of points on the right side grow leftwards
        ax.annotate(r.name, (r["n"], r["ratio"]), xytext=(-6 if right else 6, 5), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="right" if right else "left")
    tidy(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Slower than expected"),
                       Line2D([], [], marker="o", ls="", color=AQUA, label="Faster than expected"),
                       Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.6, label="Within normal variation"),
                       Line2D([], [], color=INK2, ls=(0, (4, 3)), label="±3 standard errors")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5, ncol=2)
    fig.subplots_adjust(left=0.13, right=0.97)
    ns, nf = int((sites.flag == "slower").sum()), int((sites.flag == "faster").sum())
    heading(fig, f"{ns} of {n_all} sites are credibly slower than comparable cases, {nf} faster",
            f"{label}. Each dot is a site; coloured dots pass the 5% false-discovery filter. Small sites scatter widely by chance.")
    save(fig, out, "fig_a_site_funnel")


def fig_countries(ct, out, label, n_slow=10, n_fast=5):
    t = ct.sort_values("ratio", ascending=False)
    top = t.head(n_slow)
    bot = t.tail(n_fast)
    bot = bot[~bot.index.isin(top.index)]
    d = pd.concat([top, bot])
    if d.empty:
        return
    gap = len(top) > 0 and len(bot) > 0
    ypos = list(range(len(top)))[::-1] + ([len(bot) - i - 1 - 1.0 for i in range(len(bot))] if gap else [])
    ypos = np.array(ypos[:len(top)] + [-1.4 - i for i in range(len(bot))], float)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(d) + 2.4))
    for (key, r), y in zip(d.iterrows(), ypos):
        col = BLUE if r.flag == "slower" else AQUA if r.flag == "faster" else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.8, solid_capstyle="round")
        ax.plot(r.ratio, y, "o", ms=7, color=col, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(max(r.ci_hi * 1.06, 1.15), y, f"×{r.ratio:.2f}  ·  n={int(r.n):,}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(d.index), fontsize=8.5, color=INK2)
    ratio_axis(ax, "x")
    ax.set_xlim(0.3, max(4.5, d["ci_hi"].max() * 1.6))
    ax.set_xlabel("Time vs comparable cases elsewhere (95% interval)")
    tidy(ax, "x")
    fig.subplots_adjust(left=0.2, right=0.97)
    ns = int((ct.flag == "slower").sum())
    heading(fig, f"{ns} of {len(ct)} countries are credibly slower than comparable cases",
            f"{label}. Slowest {len(top)} and fastest {len(bot)} among countries with enough cases. Blue/green = passes the 5% false-discovery filter.",
            top=0.86)
    save(fig, out, "fig_b_countries")


def fig_variance(rows, out):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    cols = [BLUE] * (len(rows) - 1) + [MUTED]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, height=0.55, color=cols, edgecolor=SURFACE)
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.1%}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK2)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of the variation in case duration")
    tidy(ax, "x")
    loc = sum(vals[2:4])
    heading(fig, f"Country and site explain about {loc:.0%} of the variation in case duration",
            "Added one step after the other, adjusted for the number of groups. Most variation is between cases at the same site.", top=0.84)
    fig.subplots_adjust(left=0.38, right=0.97)
    save(fig, out, "fig_c_variance")


def fig_pareto(opp, out):
    o = opp.sort_values(ascending=False)
    tot = o.sum()
    if tot <= 0:
        return
    xs = np.arange(1, len(o) + 1) / len(o)
    ys = o.cumsum() / tot
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(np.r_[0, xs], np.r_[0, ys], color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (4, 3)))
    i10 = max(int(round(0.10 * len(o))), 1)
    s10 = float(ys.iloc[i10 - 1])
    ax.plot([i10 / len(o)], [s10], "o", ms=7, color=BLUE, mec=SURFACE, mew=2)
    ax.annotate(f"top 10% of sites\n= {s10:.0%} of the opportunity", (i10 / len(o), s10), xytext=(22, -34),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Sites, ranked from largest to smallest opportunity")
    ax.set_ylabel("Cumulative share of the opportunity")
    tidy(ax)
    heading(fig, f"The top 10% of sites hold {s10:.0%} of the avoidable days",
            "Avoidable days = days the slower-than-expected sites lose compared with comparable cases (shrunk estimates). Dashed line: spread evenly.")
    save(fig, out, "fig_d_concentration")


def fig_within(sites, studies, out):
    d = sites[sites["Study"].isin(studies)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(studies) + 2.3))
    rng = np.random.default_rng(0)
    for i, st in enumerate(studies[::-1]):
        r = d[d["Study"] == st]["ratio"]
        ax.scatter(r, i + rng.uniform(-0.17, 0.17, len(r)), s=20, color=BLUE, alpha=0.65, edgecolors=SURFACE, linewidths=0.5)
        ax.plot([r.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
    ax.axvline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(studies)))
    nsit = d.groupby("Study").size()
    ax.set_yticklabels([f"{s}  (n={nsit[s]} sites)" for s in studies[::-1]], fontsize=8.5, color=INK2)
    ratio_axis(ax, "x")
    ax.set_xlabel("Site time vs comparable cases (case-mix adjusted; black tick = study median)")
    tidy(ax, "x")
    sp = d.groupby("Study")["ratio"].agg(lambda x: x.quantile(.9) / x.quantile(.1))
    heading(fig, f"Inside one study, the slowest tenth of sites take up to {sp.max():.1f}x longer than the fastest tenth",
            "Each dot is a site with enough cases. Sites of one study share the protocol, so gaps point to country, site or case factors.",
            top=0.86)
    fig.subplots_adjust(left=0.25, right=0.97)
    save(fig, out, "fig_e_within_study")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--type", choices=["both", "Issue", "ProtocolDeviation"], default="both")
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--min-country", type=int, default=100)
    ap.add_argument("--min-study", type=int, default=100)
    ap.add_argument("--pool-country", choices=["auto", "yes", "no"], default="auto")
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    out = d / ("rq5" if a.type == "both" else f"rq5_{a.type}")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "rq5_report.txt")

    p = d / "kpi" / "cases_kpi.csv"
    if not p.exists():
        sys.exit("kpi/cases_kpi.csv not found - run 03_kpis.py first (use the latest version).")
    cases = pd.read_csv(p, low_memory=False)
    if a.type != "both":
        cases = cases[cases["CASE_TYPE"] == a.type]
    for c in ("Study", "StudyCountry", "StudySite"):
        cases[c] = cases[c].replace("", np.nan)
    cases["Category"] = cases["Category"].fillna("(none)")
    cases["SEV_GROUP"] = cases["SEV_GROUP"].fillna("other/unknown")
    cases["COUNTRY"] = cases["StudyCountry"].str.extract(r"-(C\d+)$")[0]
    res = cases[cases["DAYS_E2E"].notna()].copy()
    res["LOGD"] = np.log1p(res["DAYS_E2E"])
    print(f"cases {len(cases):,}; resolved with a valid duration {len(res):,}; type filter: {a.type}")

    mix = [["CASE_TYPE", "SEV_GROUP", "Category"], ["CASE_TYPE", "SEV_GROUP"], ["CASE_TYPE"], []]
    res["RESID"] = res["LOGD"] - hier_expected(res, "LOGD", mix)
    sm = res.groupby("Study")["RESID"].agg(["size", "mean"])
    big = sm[sm["size"] >= a.min_study].index
    res["RESID_WS"] = np.where(res["Study"].isin(big), res["RESID"] - res["Study"].map(sm["mean"]), np.nan)
    sd_all = res["RESID"].std()

    # ---------------- R1 coverage ----------------
    h("R1  What do we have? Coverage, group sizes and a data-driven minimum")
    print(f"Study families {res['Study'].str[:5].nunique()} | studies {res['Study'].nunique()} | study-countries "
          f"{res['StudyCountry'].nunique()} | sites {res['StudySite'].nunique()} | C-numbers {res['COUNTRY'].nunique()}")
    res["LOC"] = np.where(res["StudySite"].notna(), "site known", np.where(res["StudyCountry"].notna(), "country only", "no location"))
    loc = res.groupby("LOC").agg(cases=("RESID", "size"), mean_resid=("RESID", "mean"), median_days=("DAYS_E2E", "median"),
                                 share_over_30d=("DAYS_E2E", lambda x: (x > LONG).mean()))
    loc["share_of_cases"] = loc["cases"] / loc["cases"].sum()
    loc["ratio_vs_comparable"] = np.exp(loc["mean_resid"])
    loc["ci_lo"] = np.exp(loc["mean_resid"] - 1.96 * sd_all / np.sqrt(loc["cases"]))
    loc["ci_hi"] = np.exp(loc["mean_resid"] + 1.96 * sd_all / np.sqrt(loc["cases"]))
    print("\nCases without a site or country (Novo: expected, some cases are handled centrally):")
    print(loc.drop(columns="mean_resid").round(3).to_string())
    loc.to_csv(out / "r1_location_status.csv")

    sub = res.dropna(subset=["StudySite"])
    cps = sub.groupby("StudySite").size()
    print("\nCases per site:", cps.describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_dict())
    rows = []
    for t in (10, 20, 30, 50, 100):
        g, tau = group_effects(sub, "StudySite", "RESID", t)
        rows.append({"min_cases_per_site": t, "sites_tested": len(g), "share_of_cases_covered": cps[cps >= t].sum() / cps.sum(),
                     "flagged_slower": int((g["flag"] == "slower").sum()) if len(g) else 0,
                     "flagged_faster": int((g["flag"] == "faster").sum()) if len(g) else 0,
                     "spread_sd_of_true_site_effect": tau})
    sens = pd.DataFrame(rows)
    print("\nSensitivity to the minimum size (Novo: choose it from the distribution):")
    print(sens.round(3).to_string(index=False))
    sens.to_csv(out / "r1_min_size_sensitivity.csv", index=False)

    # country numbering check
    t = res.dropna(subset=["StudyCountry"]).copy()
    t["cno"] = t["StudyCountry"].str.extract(r"-C(\d+)$")[0].astype(float)
    flags_c = []
    for st, s in t.groupby("Study"):
        u = s["cno"].dropna().unique()
        if len(u) >= 3:
            flags_c.append(u.max() == len(u))
    share_contig = float(np.mean(flags_c)) if flags_c else float("nan")
    verdict = ("study-scoped (numbered 1..k inside each study)" if share_contig >= 0.8
               else "looks global (gaps in the numbering)" if share_contig <= 0.4 else "unclear")
    print(f"\nCountry numbering check: {share_contig:.0%} of {len(flags_c)} studies have contiguous C-numbers 1..k -> {verdict}.")
    pool = a.pool_country == "yes" or (a.pool_country == "auto" and share_contig <= 0.4)
    print("Pooling C-numbers across studies:", "YES" if pool else "NO (set --pool-country yes to force it)")

    # ---------------- R2 variance ----------------
    h("R2  Where does the variation come from? (share of variation in log duration, adjusted for group count)")
    y = res["LOGD"]
    alone = []
    for lab, col in (("Case type", "CASE_TYPE"), ("Severity", "SEV_GROUP"), ("Category", "Category")):
        alone.append((lab, *adj_r2(y, res[col])))
    mixkey = res["CASE_TYPE"] + "|" + res["SEV_GROUP"] + "|" + res["Category"]
    a0, k0 = adj_r2(y, mixkey)
    ckey = res["StudyCountry"].fillna(res["Study"] + "|no-country")
    skey = res["StudySite"].fillna(ckey + "|no-site")
    r = res["RESID"]
    a1, k1 = adj_r2(r, res["Study"])
    a2, k2 = adj_r2(r, ckey)
    a3, k3 = adj_r2(r, skey)
    rest = 1 - a0
    steps = [("Case mix (type, severity, category)", a0), ("Study (given case mix)", rest * a1),
             ("Country within study", rest * max(a2 - a1, 0)), ("Site within country", rest * max(a3 - a2, 0))]
    steps.append(("Not explained (differences between cases)", max(0.0, 1 - sum(v for _, v in steps))))
    for lab, v, k in alone:
        print(f"  {lab:<28} alone: {v:6.1%}  ({k} groups)")
    print(f"  Case mix combined ({k0} groups): {a0:.1%}")
    for lab, v in steps:
        print(f"  {lab:<40} {v:6.1%}")
    pd.DataFrame(steps, columns=["step", "share_of_variation"]).to_csv(out / "r2_variance.csv", index=False)
    print("(Sites are nested in countries, countries in studies, so each step adds what the finer level explains on top.)")

    # ---------------- R3 sites and countries ----------------
    h("R3  Site and country ranking, case-mix adjusted ('apples to apples')")
    rs = res.dropna(subset=["StudySite"])
    sites, tau_s = group_effects(rs, "StudySite", "RESID", a.min_site)
    sites_ws, tau_s_ws = group_effects(rs.dropna(subset=["RESID_WS"]), "StudySite", "RESID_WS", a.min_site)
    raw = rs.groupby("StudySite")["DAYS_E2E"].agg(median_days="median", share_over_30d=lambda x: (x > LONG).mean(),
                                                  days_sum="sum")
    sites = sites.join(raw)
    sites["opp_days"] = sites["days_sum"] * (1 - np.exp(-sites["m_shrunk"].clip(lower=0)))
    sites["Study"], sites["StudyCountry"] = key_parts(pd.Series(sites.index, index=sites.index))
    sites = sites.join(sites_ws[["ratio", "ci_lo", "ci_hi", "q", "flag"]].add_suffix("_ws"))
    sites.index.name = "StudySite"
    sites.to_csv(out / "r3_sites.csv")
    print(f"Sites tested (>= {a.min_site} cases): {len(sites)} | credibly slower {int((sites.flag == 'slower').sum())} | "
          f"credibly faster {int((sites.flag == 'faster').sum())}  (FDR {FDR:.0%})")
    print(f"Spread between true site effects after case mix: SD {tau_s:.2f} (log) -> a typical site is about "
          f"{(math.exp(tau_s) - 1):.0%} above or below comparable cases.")
    print(f"Compared only with sites of the SAME study: SD {tau_s_ws:.2f} -> about {(math.exp(tau_s_ws) - 1):.0%}.")
    print(f"Sites flagged slower in BOTH views: {int(((sites.flag == 'slower') & (sites.flag_ws == 'slower')).sum())}")
    cols = ["n", "median_days", "share_over_30d", "ratio", "ci_lo", "ci_hi", "q", "ratio_shrunk", "ratio_ws", "flag"]
    print("\nSlowest sites by shrunk ratio (small sites are pulled toward 1, so this ranking is robust):")
    show(sites, cols, 10, "ratio_shrunk")
    print("\nFastest sites (candidates to learn from):")
    show(sites, cols, 6, "ratio_shrunk", True)
    print("\nNote: Group 1's example sites P0008-T0001-C0012-S0043 and P0008-T0001-C0005-S0008 can be looked up in r3_sites.csv.")

    ctab = {}
    for label, key, mn in (("study-country", "StudyCountry", a.min_country), ("country (C-number pooled across studies)", "COUNTRY", a.min_country)):
        if key == "COUNTRY" and not pool:
            continue
        rc = res.dropna(subset=[key])
        ct, tau = group_effects(rc, key, "RESID", mn)
        if ct.empty:
            print(f"\n{label}: no group has {mn}+ cases")
            continue
        ct = ct.join(rc.groupby(key)["DAYS_E2E"].agg(median_days="median", share_over_30d=lambda x: (x > LONG).mean()))
        ct.index.name = key
        ct.to_csv(out / f"r3_{'countries_pooled' if key == 'COUNTRY' else 'study_countries'}.csv")
        ctab[key] = ct
        print(f"\n{label}: {len(ct)} tested (>= {mn} cases) | slower {int((ct.flag == 'slower').sum())} | "
              f"faster {int((ct.flag == 'faster').sum())} | spread SD {tau:.2f} -> about {(math.exp(tau) - 1):.0%}")
        show(ct, ["n", "median_days", "share_over_30d", "ratio", "ci_lo", "ci_hi", "q", "flag"], 8, "ratio")
        ws, tau_w = group_effects(rc.dropna(subset=["RESID_WS"]), key, "RESID_WS", mn)
        if len(ws):
            print(f"  same-study comparison: slower {int((ws.flag == 'slower').sum())}, faster {int((ws.flag == 'faster').sum())}, spread SD {tau_w:.2f}")

    # ---------------- R4 concentration ----------------
    h("R4  How concentrated is the opportunity?")
    opp = sites["opp_days"]
    tot_opp = float(opp.sum())
    exc30 = float((res["DAYS_E2E"] - LONG).clip(lower=0).sum())
    tot_days = float(res["DAYS_E2E"].sum())
    o = opp.sort_values(ascending=False)
    print(f"Avoidable days if every slower-than-expected site matched comparable cases (shrunk, approximate): {tot_opp:,.0f}")
    print(f"  = {tot_opp / tot_days:.1%} of all case-days and {tot_opp / exc30:.1%} of the days beyond {LONG}. "
          f"Covers only the {len(sites)} sites with {a.min_site}+ cases, so it is a conservative figure.")
    for pc in (0.05, 0.10, 0.20):
        k_ = max(int(round(pc * len(o))), 1)
        print(f"  top {pc:.0%} of sites ({k_}) hold {o.iloc[:k_].sum() / tot_opp:.0%} of it")
    n50 = int((o.cumsum() / tot_opp < 0.5).sum() + 1)
    print(f"  {n50} sites hold half of it")

    # ---------------- R5 within-study ----------------
    h("R5  Within the same study: how far apart are the sites? (protocol and therapeutic area held constant)")
    g = sites.groupby("Study")["ratio"].agg(n_sites="size", fastest=lambda x: x.quantile(.1), slowest=lambda x: x.quantile(.9))
    g["spread_p90_over_p10"] = g["slowest"] / g["fastest"]
    g["cases"] = sites.groupby("Study")["n"].sum()
    g = g[g["n_sites"] >= 5].sort_values("cases", ascending=False)
    g.to_csv(out / "r5_within_study.csv")
    show(g, ["n_sites", "cases", "fastest", "slowest", "spread_p90_over_p10"], 12, "cases")
    print(f"Median spread between the slowest and fastest tenth of sites inside a study: x{g['spread_p90_over_p10'].median():.1f}"
          if len(g) else "no study has 5+ tested sites")

    # ---------------- R6 rework ----------------
    h("R6  Rework by site and country (reopened cases, observed vs expected for the case mix)")
    res["REOPENED"] = (res["N_REOPEN"] > 0).astype(int)
    res["P_RE"] = hier_expected(res, "REOPENED", mix)
    rs = res.dropna(subset=["StudySite"])
    re_s = group_binary(rs, "StudySite", "REOPENED", "P_RE", a.min_site)
    print(f"Overall reopening rate {res['REOPENED'].mean():.1%}. Sites tested {len(re_s)} | more than expected "
          f"{int((re_s.flag == 'slower').sum()) if len(re_s) else 0} | fewer {int((re_s.flag == 'faster').sum()) if len(re_s) else 0}")
    show(re_s, ["n", "rate", "ratio", "ci_lo", "ci_hi", "q", "flag"], 8, "ratio")
    re_s.to_csv(out / "r6_reopen_sites.csv")
    re_c = group_binary(res.dropna(subset=["StudyCountry"]), "StudyCountry", "REOPENED", "P_RE", a.min_country)
    print(f"\nStudy-countries tested {len(re_c)} | more than expected {int((re_c.flag == 'slower').sum()) if len(re_c) else 0}")
    show(re_c, ["n", "rate", "ratio", "ci_lo", "ci_hi", "q", "flag"], 6, "ratio")
    re_c.to_csv(out / "r6_reopen_countries.csv")
    rho, nn = spearman(sites["m"], np.log(re_s["ratio"].replace(0, np.nan)) if len(re_s) else pd.Series(dtype=float))
    print(f"\nDo sites that are slower also reopen more? Spearman rho = {rho:.2f} over {nn} sites")

    # ---------------- R7 PD-only: registration lag and review cycles ----------------
    h("R7  Protocol deviations only: registration lag and review cycles by site")
    mp = d / "model" / "cases_pd.csv"
    lag_s = None
    if mp.exists() and a.type != "Issue":
        need = {"ID", "DateIdentified", "CreatedDate", "NoOfReviewCycles"}
        pdm = pd.read_csv(mp, usecols=lambda c: c in need, low_memory=False)
        pdr = res[res["CASE_TYPE"] == "ProtocolDeviation"].merge(pdm, on="ID", how="left")
        ls = [["SEV_GROUP", "Category"], ["SEV_GROUP"], []]
        if {"DateIdentified", "CreatedDate"} <= set(pdr.columns):
            lag = (pd.to_datetime(pdr["CreatedDate"], format="mixed", errors="coerce")
                   - pd.to_datetime(pdr["DateIdentified"], format="mixed", errors="coerce")).dt.total_seconds() / 86400
            neg = int((lag < 0).sum())
            q = pdr[lag >= 0].copy()
            q["LAG"] = lag[lag >= 0]
            q["LOGL"] = np.log1p(q["LAG"])
            q["RESID"] = q["LOGL"] - hier_expected(q, "LOGL", ls)
            print(f"Registration lag (DateIdentified -> CreatedDate): median {q['LAG'].median():.1f} days, "
                  f"{(q['LAG'] > 7).mean():.0%} over a week; {neg:,} cases created BEFORE identified are dropped.")
            lag_s, tau_l = group_effects(q.dropna(subset=["StudySite"]), "StudySite", "RESID", a.min_site)
            if len(lag_s):
                lag_s = lag_s.join(q.groupby("StudySite")["LAG"].median().rename("median_lag_days"))
                print(f"Sites tested {len(lag_s)} | slower to register {int((lag_s.flag == 'slower').sum())} | faster "
                      f"{int((lag_s.flag == 'faster').sum())} | spread SD {tau_l:.2f} -> about {(math.exp(tau_l) - 1):.0%}")
                show(lag_s, ["n", "median_lag_days", "ratio", "ci_lo", "ci_hi", "q", "flag"], 6, "ratio")
                lag_s.to_csv(out / "r7_registration_lag_sites.csv")
        if "NoOfReviewCycles" in pdr.columns:
            pdr["RETURNED"] = (pd.to_numeric(pdr["NoOfReviewCycles"], errors="coerce") >= 2).astype(int)
            pdr["P_RET"] = hier_expected(pdr, "RETURNED", ls)
            rv = group_binary(pdr.dropna(subset=["StudySite"]), "StudySite", "RETURNED", "P_RET", a.min_site)
            print(f"\nMore than one review cycle (NoOfReviewCycles >= 2; my assumed definition of a review return): "
                  f"{pdr['RETURNED'].mean():.1%} of PDs.")
            if len(rv):
                print(f"Sites tested {len(rv)} | more returns than expected {int((rv.flag == 'slower').sum())} | fewer {int((rv.flag == 'faster').sum())}")
                show(rv, ["n", "rate", "ratio", "ci_lo", "ci_hi", "q", "flag"], 6, "ratio")
                rv.to_csv(out / "r7_review_cycles_sites.csv")
    else:
        print("model/cases_pd.csv not found or Issues only - skipped")

    # ---------------- R8 tasks and follow-ups ----------------
    h("R8  Do the same sites also have slow tasks and follow-ups?")
    site_task = {}
    for name, f in (("tasks", "tasks.csv"), ("follow-ups", "tms.csv")):
        mp = d / "model" / f
        if not mp.exists():
            print(f"{f} not found - skipped")
            continue
        tk = pd.read_csv(mp, usecols=lambda c: c in {"Site", "StudyCountry", "Study", "Category", "DAYS_CREATED_TO_COMPLETED"},
                         low_memory=False)
        tk = tk[tk["DAYS_CREATED_TO_COMPLETED"] >= 0].dropna(subset=["Site"]).copy()
        tk["Category"] = tk["Category"].fillna("(none)")
        tk["LOGD"] = np.log1p(tk["DAYS_CREATED_TO_COMPLETED"])
        tk["RESID"] = tk["LOGD"] - hier_expected(tk, "LOGD", [["Category"], []])
        ts, tau_t = group_effects(tk, "Site", "RESID", a.min_site)
        if ts.empty:
            print(f"{name}: no site has {a.min_site}+ items")
            continue
        ts = ts.join(tk.groupby("Site")["DAYS_CREATED_TO_COMPLETED"].median().rename("median_days"))
        ts.to_csv(out / f"r8_{name.replace('-', '')}_sites.csv")
        site_task[name] = ts["m"]
        print(f"\n{name}: {len(tk):,} items | sites tested {len(ts)} | slower {int((ts.flag == 'slower').sum())} | "
              f"faster {int((ts.flag == 'faster').sum())} | spread SD {tau_t:.2f} -> about {(math.exp(tau_t) - 1):.0%}")
        show(ts, ["n", "median_days", "ratio", "ci_lo", "ci_hi", "q", "flag"], 6, "ratio")
        rho, nn = spearman(sites["m"], ts["m"])
        print(f"Sites slow on cases vs slow on {name}: Spearman rho = {rho:.2f} over {nn} sites")
        if len(ts) < 30:    # sparse: also compare at study-country level, where there are enough items
            tcn = max(2 * a.min_site, 50)
            tc, tau_c = group_effects(tk.dropna(subset=["StudyCountry"]), "StudyCountry", "RESID", tcn)
            if len(tc):
                tc = tc.join(tk.groupby("StudyCountry")["DAYS_CREATED_TO_COMPLETED"].median().rename("median_days"))
                tc.to_csv(out / f"r8_{name.replace('-', '')}_countries.csv")
                print(f"  few sites reach {a.min_site} items, so at study-country level ({tcn}+ items): {len(tc)} tested | "
                      f"slower {int((tc.flag == 'slower').sum())} | faster {int((tc.flag == 'faster').sum())} | spread SD {tau_c:.2f}")
                show(tc, ["n", "median_days", "ratio", "ci_lo", "ci_hi", "q", "flag"], 5, "ratio")

    h("R9  Do sites that are slow on Issues also run slow on protocol deviations?  (Novo Q&A: 'both can be added')")
    if a.type == "both":
        for lab, col in (("case mix only", "RESID"), ("same-study comparison", "RESID_WS")):
            pv = rs.dropna(subset=[col]).groupby(["StudySite", "CASE_TYPE"])[col].agg(["size", "mean"]).reset_index()
            pv = pv[pv["size"] >= 20].pivot(index="StudySite", columns="CASE_TYPE", values="mean")
            if {"Issue", "ProtocolDeviation"} <= set(pv.columns):
                rho, nn = spearman(pv["Issue"], pv["ProtocolDeviation"])
                print(f"  {lab:<22} Spearman rho between a site's Issue effect and its PD effect = {rho:.2f} over {nn} sites with 20+ of each")
        print("  The case-mix-only value also contains shared study and country context; the same-study value is the cleaner site signal.")
        print("  Positive = slowness belongs to the site (people, practice); near zero = it belongs to the process type.")
    else:
        print("only meaningful with --type both")

    # ---------------- figures ----------------
    h("Figures")
    pl = out / "plots"
    label = "Resolved cases, adjusted for type, severity and category" + ("" if a.type == "both" else f" ({a.type} only)")
    jobs = [
        lambda: fig_funnel(sites, pl, label),
        lambda: fig_countries(ctab.get("COUNTRY", ctab.get("StudyCountry")), pl,
                              "Pooled countries" if "COUNTRY" in ctab else "Study-countries (a country inside one study)"),
        lambda: fig_variance(steps, pl),
        lambda: fig_pareto(opp, pl),
        lambda: fig_within(sites, list(g.sort_values("n_sites", ascending=False).head(8).index), pl),
    ]
    for j in jobs:
        try:
            j()
        except Exception as e:  # one broken figure must not block the others
            print(f"figure FAILED: {type(e).__name__}: {e}")
    print(f"\nDone. Files in {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""sl_figs_en.py -- English versions of figs 23/24/25 for the Interface paper.

Same model (sl.py), same seeds and configurations as sl_experiments.py and
sl_compare.py. Only the labels and the notation change: w -> a (attention
gate). NOTE: random streams differ between NumPy versions, so values here
(NumPy 2.4.3) differ slightly from the author's original JSONs; the paper's
tables are regenerated from THIS script's output (sl_tables_en.json).
Outputs: figs/fig23_rogers_en.{pdf,png}, fig24_drift_test_en, fig25_compare_en.
"""
import os, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sl

TAB = {}
NS = 12
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .3, "grid.linewidth": .5,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "figure.constrained_layout.use": True})

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", red="#e34948",
         gray="#898781", ink="#0b0b0b", grid="#e1e0d9", violet="#4a3aa7",
         green="#008300")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------ fig23: Rogers' paradox
def _rogers_evolved(s):
    return sl.run(generations=400, seed=5000 + s, evolve_w=True,
                  who="top_decile", k=5)

def _rogers_fixed(arg):
    w, s = arg
    return (w, sl.run(generations=400, seed=5100 + s, w=w,
                      who="top_decile", k=5)["fmean"][-100:].mean())

def fig_rogers(ex):
    rr = list(ex.map(_rogers_evolved, range(NS)))
    corr = float(np.mean([np.nanmean(r["wF_corr"][-100:]) for r in rr]))
    hi = float(np.mean([np.nanmean(r["F_hiw"][-100:]) for r in rr]))
    lo = float(np.mean([np.nanmean(r["F_low"][-100:]) for r in rr]))
    aev = float(np.mean([r["w_mean"][-1] for r in rr]))
    Fev = float(np.mean([r["fmean"][-100:].mean() for r in rr]))
    ws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    res = list(ex.map(_rogers_fixed, [(w, s) for w in ws for s in range(NS)]))
    Ffix = [float(np.mean([f for w2, f in res if w2 == w])) for w in ws]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].bar([0, 1], [lo, hi], 0.55, color=[C["gray"], C["blue"]])
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["lowest quartile of $a$", "highest quartile of $a$"])
    ax[0].set_ylim(min(lo, hi) - 0.35, max(lo, hi) + 0.12)
    ax[0].set_ylabel("individual fitness")
    ax[0].set_title(f"Within a population, a more open gate pays:\n"
                    f"correlation of $a$ with fitness $= {corr:+.2f}$", loc="left")
    ax[1].plot(ws, Ffix, "-o", ms=4, color=C["red"], label="$a$ held fixed")
    ax[1].scatter([aev], [Fev], s=70, marker="*", color=C["blue"], zorder=5,
                  label="$a$ left free to evolve")
    ax[1].set_xlabel("attention gate $a$")
    ax[1].set_ylabel("population mean fitness")
    ax[1].legend(fontsize=7.5, loc="lower left")
    ax[1].set_title("Across populations, a more open gate does not pay", loc="left")
    save(fig, "fig23_rogers_en")
    TAB["rogers"] = dict(corr=round(corr, 3), F_hi=round(hi, 3), F_lo=round(lo, 3),
                         a_evolved=round(aev, 3), F_evolved=round(Fev, 3),
                         F_fixed={str(w): round(f, 3) for w, f in zip(ws, Ffix)})
    print(TAB["rogers"])


# ------------------------------------------------ fig24: drift test
def _drift_base(s):
    return sl.run(generations=3000, seed=6000 + s, evolve_w=True,
                  who="top_decile", k=5)["loci_q"][-1]

def _drift_N(arg):
    N, s = arg
    return (N, sl.run(generations=3000, N=N, seed=7000 + s, evolve_w=True,
                      who="top_decile", k=5)["pq"][-1])

def fig_drift(ex):
    lq = np.concatenate(list(ex.map(_drift_base, range(6))))
    Ns = [125, 250, 500, 1000, 2000]
    res = list(ex.map(_drift_N, [(N, s) for N in Ns for s in range(4)]))
    pq = [float(np.mean([v for N2, v in res if N2 == N])) for N in Ns]
    b = float(np.polyfit(np.log(Ns), np.log(pq), 1)[0])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].hist(lq, bins=np.linspace(0, 1, 26), color=C["blue"])
    ax[0].set_xlabel("frequency of '?' at a locus, generation 3000")
    ax[0].set_ylabel("number of loci (6 runs $\\times$ 20)")
    ax[0].set_title("Two spikes, nothing between:\ndrift, not a selective equilibrium",
                    loc="left")
    ax[1].plot(Ns, pq, "-o", ms=4, color=C["red"],
               label=f"slope $\\approx {b:.2f}$ (log--log)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("population size $N$")
    ax[1].set_ylabel("final frequency of '?'")
    ax[1].legend(fontsize=7.5)
    ax[1].set_title("Retained plasticity falls with $N$\n(4 runs per point)",
                    loc="left")
    save(fig, "fig24_drift_test_en")
    TAB["drift"] = dict(fixed0=round(float((lq < .01).mean()), 3),
                        stuck=round(float((lq > .99).mean()), 3),
                        mid=round(float(((lq > .15) & (lq < .85)).mean()), 3),
                        slope=round(b, 3),
                        N_sweep={str(n): round(x, 3) for n, x in zip(Ns, pq)})
    print(TAB["drift"])


# ------------------------------------------------ fig25: seven regimes
ROWS = [
 ("evolution alone",            dict(plastic=False, w=0.0),                 C["gray"]),
 ("individual learning, $a{=}0$", dict(w=0.0, who='parent'),                C["ink"]),
 ("social alone, $k{=}1$ parent", dict(w=1.0, who='parent'),                C["red"]),
 ("social alone, $k{=}5$, fittest tenth", dict(w=1.0, k=5, who='top_decile'),       C["orange"]),
 ("blend, $a{=}0.8$",            dict(w=0.8, k=5, who='top_decile'),        C["blue"]),
 ("blend, $a$ evolves",          dict(evolve_w=True, k=5, who='top_decile'), C["aqua"]),
 ("switch",            dict(mode='switch', who='parent'),          C["violet"]),
]
LONG = [("individual learning", dict(w=0.0, who='parent')),
        ("switch", dict(mode='switch', who='parent')),
        ("blend, $a$ evolves", dict(evolve_w=True, k=5, who='top_decile')),
        ("blend, $a{=}0.8$", dict(w=0.8, k=5, who='top_decile'))]

def _row(arg):
    i, s = arg
    r = sl.run(generations=200, seed=1000 + s, **ROWS[i][1])
    nw = r["n_winner"]
    first = int(np.argmax(nw > 0)) if (nw > 0).any() else -1
    return (i, r["fmean"][-1], r["pq"][-1], r["p1"][-1], first)

def _long(arg):
    i, s = arg
    r = sl.run(generations=2500, seed=9000 + s, **LONG[i][1])
    return (i, r["loci_q"][-1], float(r["pq"][-1]))

def fig_compare(ex):
    res = list(ex.map(_row, [(i, s) for i in range(len(ROWS)) for s in range(NS)]))
    pts = []
    for i in range(len(ROWS)):
        rr = [x for x in res if x[0] == i]
        F = float(np.mean([x[1] for x in rr])); q = float(np.mean([x[2] for x in rr]))
        one = float(np.mean([x[3] for x in rr]))
        solved = int(sum(1 for x in rr if x[4] >= 0))
        gens = [x[4] for x in rr if x[4] >= 0]
        TAB.setdefault("r5compare", {})[ROWS[i][0]] = dict(
            F=round(F, 2), q=round(q, 3), one=round(one, 3), solved=solved,
            gen=int(np.median(gens)) if gens else -1)
        pts.append((F, q))
    resL = list(ex.map(_long, [(i, s) for i in range(len(LONG)) for s in range(6)]))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for (lab, kw, col), (F, q) in zip(ROWS, pts):
        ax[0].scatter(q, F, s=42, color=col, zorder=5)
    # label layout: relative offsets for isolated points; absolute positions
    # with leader lines for the overlapping cluster at the top
    off = {"evolution alone": (0.03, 0.0),
           "individual learning, $a{=}0$": (-0.09, 1.8),
           "social alone, $k{=}1$ parent": (0.03, 0.0),
           "social alone, $k{=}5$, fittest tenth": (0.74, 14.0),
           "blend, $a{=}0.8$": (0.74, 17.2),
           "blend, $a$ evolves": (0.56, 22.2),
           "switch": (0.70, 21.0)}
    cluster = {"social alone, $k{=}5$, fittest tenth", "blend, $a{=}0.8$",
               "blend, $a$ evolves", "switch"}
    for (lab, kw, col), (F, q) in zip(ROWS, pts):
        xt, yt = off[lab]
        if lab in cluster:
            ax[0].annotate(lab, (q, F), xytext=(xt, yt), fontsize=7, color=col,
                           ha="left", va="center",
                           arrowprops=dict(arrowstyle="-", lw=0.6, color=col,
                                           shrinkA=1, shrinkB=2))
        else:
            ax[0].annotate(lab, (q, F), xytext=(q + xt, F + yt),
                           fontsize=7, color=col, ha="left", va="center")
    ax[0].set_xlabel("frequency of '?' at generation 200")
    ax[0].set_ylabel("population mean fitness")
    ax[0].set_xlim(-0.04, 1.02); ax[0].set_ylim(0, 23.5)
    ax[0].set_title("Seven regimes, one landscape", loc="left")

    labs = ["individual\nlearning", "switch", "blend,\n$a$ evolves",
            "blend,\n$a{=}0.8$"]
    free = []; stuck = []; mid = []
    for i in range(len(LONG)):
        lq = np.concatenate([x for i2, x, pq in resL if i2 == i])
        pqm = float(np.mean([pq for i2, x, pq in resL if i2 == i]))
        free.append(float((lq < .01).mean())); stuck.append(float((lq > .99).mean()))
        mid.append(1 - free[-1] - stuck[-1])
        TAB.setdefault("r5long", {})[LONG[i][0]] = dict(
            pq=round(pqm, 3), free=round(free[-1], 3), stuck=round(stuck[-1], 3),
            mid=round(mid[-1], 3))
    x = np.arange(len(LONG))
    ax[1].bar(x, free, 0.6, color=C["gray"], label="fixed, no ?")
    ax[1].bar(x, stuck, 0.6, bottom=free, color=C["red"], label="stuck at ?")
    ax[1].bar(x, mid, 0.6, bottom=np.array(free) + stuck, color=C["aqua"],
              label="intermediate")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=7)
    ax[1].set_ylabel("fraction of loci")
    ax[1].set_title("Locus composition after 2500 generations", loc="left")
    ax[1].legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig25_compare_en")
    print(TAB["r5compare"]); print(TAB["r5long"])


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=16) as ex:
        fig_compare(ex)
        fig_rogers(ex)
        fig_drift(ex)
    json.dump(TAB, open(os.path.join(FIG, "..", "sl_tables_en.json"), "w"), indent=1)
    print("DONE")

# -*- coding: utf-8 -*-
"""hn_periph8_fig.py -- hinh trung tam: dau cua chon loc len plasticity tren
mat phang (a, psi), va duong bien giai tich.

Tieu chuan dau (giai tich, cung chuoi Belew/Fontanari):
  E[F | q] = (1 - psi (L-q)/L) * F_learn(q) + psi (L-q)/L * F_min
  F_learn(q) = 1 + 19 (1 - E[g]/G),  E[g] = (1-(1-P)^G)/P,  P = p^q,
  p = D_b( a*c + (1-a)/2 ),  c = f  (cue tai dich, gioi han boi fidelity)
Dau cua d E[F]/dq tai q = q_bar cua o:  duong = chon loc uu ai plasticity.
  - psi chi xuat hien trong so hang LOI (moi truong doi -> loci co dinh sai)
  - a, b chi xuat hien trong p, tuc so hang CHI PHI cua viec de mo
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
import hn_periph as hp

L, G, F_MIN, F_MAX, FID = 20, 1000, 1.0, 20.0, 0.9
HERE = os.path.dirname(os.path.abspath(__file__))
ST = json.load(open(os.path.join(HERE, "..", "hn_periph8_stats.json")))
A = sorted({c['a'] for c in ST}); PSI = sorted({c['psi'] for c in ST}); B = sorted({c['b'] for c in ST})

def F_learn(q, p):
    P = p ** q
    Eg = (1 - (1 - P) ** G) / P if P > 0 else G
    return 1 + 19 * max(0.0, 1 - Eg / G)

def dEF_dq(a, b, psi, q):
    """finite difference of E[F|q] at q (q -> q+1)."""
    p = float(hp.D(a * FID + (1 - a) / 2, b))
    def EF(qq):
        hit = psi * (L - qq) / L
        return (1 - hit) * F_learn(qq, p) + hit * F_MIN
    return EF(q + 1) - EF(q)

def psi_star(a, b, q):
    """smallest psi with dE[F]/dq >= 0 at this q (None if never)."""
    p = float(hp.D(a * FID + (1 - a) / 2, b))
    cost = F_learn(q, p) - F_learn(q + 1, p)            # >= 0
    gain = (F_learn(q + 1, p) - F_MIN) / L               # per unit psi, from (L-q)->(L-q-1)
    # dEF/dq = -(1-psi(L-q)/L)*cost ... solve exactly by scan
    for ps in np.linspace(0, 1, 2001):
        if dEF_dq(a, b, ps, q) >= 0: return float(ps)
    return None

# ---- predicted vs observed sign per surviving cell ----
agree = total = 0; rows = []
for c in ST:
    if c['n_bq'] < 6:  # need enough seeds to call a sign
        c['pred'] = None; continue
    q = int(round(L * c['pq_mean']))
    q = min(max(q, 1), L - 1)
    d = dEF_dq(c['a'], c['b'], c['psi'], q)
    c['pred'] = d >= 0
    obs = c['frac_pos'] > 0.5
    c['obs'] = obs; c['q'] = q; c['dEF'] = d
    total += 1; agree += (c['pred'] == obs)
    rows.append((c['b'], c['a'], c['psi'], q, d, c['k_pos'], c['n_bq'], c['pred'] == obs))
print(f"analytic sign criterion agrees with observed majority sign in {agree}/{total} cells (n_bq>=6)")
for r in rows:
    if not r[-1]: print("  disagree:", r)

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 9})
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
def edges(v):
    v = np.asarray(v, float); m = (v[1:] + v[:-1]) / 2
    return np.concatenate([[v[0] - (m[0] - v[0])], m, [v[-1] + (v[-1] - m[-1])]])
xe, ye = np.arange(len(PSI)+1)-0.5, np.arange(len(A))-0.5
ye = np.arange(len(A)+1)-0.5
XI = {p:i for i,p in enumerate(PSI)}; YI = {a:i for i,a in enumerate(A)}
vmax = 0.2
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, constrained_layout=True)
for ax, b in zip(axes, B):
    Z = np.full((len(A), len(PSI)), np.nan)
    for i, a in enumerate(A):
        for j, p in enumerate(PSI):
            c = next(x for x in ST if x['a'] == a and x['b'] == b and x['psi'] == p)
            if c['n_bq'] >= 6: Z[i, j] = c['eff_mean']
    ax.pcolormesh(xe, ye, Z, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax),
                  edgecolors="#fcfcfb", linewidth=1.5)
    for i, a in enumerate(A):
        for j, p in enumerate(PSI):
            c = next(x for x in ST if x['a'] == a and x['b'] == b and x['psi'] == p)
            x0, x1, y0, y1 = xe[j], xe[j + 1], ye[i], ye[i + 1]
            if c['n_bq'] < 6:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#e1e0d9",
                                       edgecolor="#fcfcfb", hatch="////", linewidth=1.5))
                txt = f"{c['n_alive']} alive" if c['n_alive'] else "extinct"
                ax.text(XI[p], YI[a], txt, ha="center", va="center", fontsize=6.3, color="#52514e")
            else:
                ax.text(XI[p], YI[a], f"{c['k_pos']}/{c['n_bq']}", ha="center", va="center",
                        fontsize=7.2, color="#0b0b0b")
                if c.get('pred'):
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                           edgecolor="#0b0b0b", linewidth=1.2))
    ax.set_xticks(range(len(PSI))); ax.set_xticklabels([f"{p:g}" for p in PSI])
    ax.set_yticks(range(len(A))); ax.set_yticklabels([f"{a:g}" for a in A])
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(r"rate of environmental change $\psi$")
    ax.set_title(f"conformity $b={b:g}$", loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel(r"attention gate $a$")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax))
cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
cb.set_label(r"selection on plasticity, $\beta_q\cdot\mathrm{SD}(n_?)$  (fitness units)")
fig.text(0.01, -0.03,
         "cell label: runs with $\\beta_q>0$ / runs measured;  black outline: analytic criterion "
         "predicts $\\beta_q>0$;  hatched: population extinct", fontsize=7, color="#52514e")
out = os.path.join(HERE, "..", "figs", "fig26_sign_boundary")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=160, bbox_inches="tight")
json.dump(dict(agree=agree, total=total,
               cells=[dict(b=r[0], a=r[1], psi=r[2], q=r[3], dEF=r[4], k=r[5], n=r[6], ok=r[7]) for r in rows]),
          open(os.path.join(HERE, "..", "hn_periph8_pred.json"), "w"), indent=1)
print("saved", out)

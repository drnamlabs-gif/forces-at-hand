# -*- coding: utf-8 -*-
"""hn_periph8_stats.py -- thong ke theo tung seed cho quet a x psi (hn_periph8).
Moi o (a, b, psi): n_alive/12; cho bq_traj (phep do tren quy dao):
  trung binh, CI bootstrap 95% (theo seed), so seed duong / n,
  sign test nhi thuc hai phia (H0: P(beta>0)=1/2), beta*SD.
"""
import json, os
from math import comb
import numpy as np

rng = np.random.default_rng(0)
res = json.load(open(os.path.join("..", "hn_periph8.json")))
A = sorted({x['a'] for x in res}); B = sorted({x['b'] for x in res})
PSI = sorted({x['psi'] for x in res})

def binom_two_sided(k, n):
    if n == 0: return float('nan')
    p = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    return min(1.0, p)

def boot_ci(x, nb=4000):
    x = np.asarray(x)
    if len(x) < 2: return [float('nan'), float('nan')]
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(nb)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

out = []
for b in B:
    for a in A:
        for p in PSI:
            r = [x for x in res if x['a'] == a and x['b'] == b and x['psi'] == p]
            al = [x for x in r if x['alive']]
            fin = [x for x in al if x['bq_traj'] == x['bq_traj']]
            bq = [x['bq_traj'] for x in fin]; eff = [x['eff_traj'] for x in fin]
            tail = [x['bq_tail'] for x in al if x['bq_tail'] == x['bq_tail']]
            k = int(sum(v > 0 for v in bq)); n = len(bq)
            out.append(dict(a=a, b=b, psi=p, n=len(r), n_alive=len(al), n_bq=n,
                            P_alive=len(al) / len(r),
                            bq_mean=float(np.mean(bq)) if n else float('nan'),
                            bq_ci=boot_ci(bq), k_pos=k,
                            frac_pos=k / n if n else float('nan'),
                            sign_p=binom_two_sided(k, n),
                            eff_mean=float(np.mean(eff)) if n else float('nan'),
                            bq_tail_mean=float(np.mean(tail)) if tail else float('nan'),
                            n_tail=len(tail),
                            pq_mean=float(np.mean([x['pq'] for x in al])) if al else float('nan')))
json.dump(out, open(os.path.join("..", "hn_periph8_stats.json"), "w"), indent=1)

for b in B:
    print(f"\n=== b={b}   a, psi : alive/12 | bq_traj [95% CI] | pos/n (sign p) | beta*SD | p(?)")
    for a in A:
        for p in PSI:
            c = next(x for x in out if x['a'] == a and x['b'] == b and x['psi'] == p)
            if c['n_bq'] == 0:
                print(f"a={a:.1f} psi={p:.2f}  {c['n_alive']:2d}/12 | extinct"); continue
            lo, hi = c['bq_ci']
            print(f"a={a:.1f} psi={p:.2f}  {c['n_alive']:2d}/12 | {c['bq_mean']:+7.3f} "
                  f"[{lo:+7.3f},{hi:+7.3f}] | {c['k_pos']:2d}/{c['n_bq']:2d} (p={c['sign_p']:.3f}) "
                  f"| {c['eff_mean']:+6.3f} | {c['pq_mean']:.3f}")
print("\nDONE")

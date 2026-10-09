# -*- coding: utf-8 -*-
"""hn_periph8.py -- QUET a x psi: dau cua beta_q do AI quyet dinh?

Cau hoi (causal specificity): van psi (moi truong) doi dau beta_q, hay van a
(cong chu y) doi dau?  Neu dau chi phu thuoc psi (mien quan the con song) thi:
  - moi truong  = nguyen nhan INSTRUCTIVE (quyet dinh huong chon loc)
  - kenh xa hoi = nguyen nhan PERMISSIVE  (can de song, khong doi huong)

Hai phep do beta_q cho moi seed:
  bq_tail : trung binh 100 the he cuoi (nhu hn_periph6; mat xac dinh khi p(?)->1)
  bq_traj : trung binh tren toan quy dao sau burn-in, chi lay the he co
            Var(n_q) >= VMIN  (dung duoc ca khi plasticity co dinh ve cuoi)
Cung model, cung tham so hn_periph6 (N=1000, L=20, G=1000, k=10, f=0.9).
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import hn_periph as hp
from hn_periph3 import L, G, FMAX, FMIN

NS, GENS, BURN, VMIN = 12, 400, 50, 0.05
A   = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PSI = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25)
B   = (0.0, 0.5)
P0  = 0.8

def run(seed, psi, a0, b0, p0, N=1000, k=10, f=0.9, n_flip=1, gens=GENS):
    rng = np.random.default_rng(seed)
    theta = np.ones(L, dtype=np.int8)
    pop = rng.choice([0, 1, 2], size=(N, L),
                     p=((1 - p0) / 2, (1 - p0) / 2, p0)).astype(np.int8)
    a = np.full(N, a0); b = np.full(N, b0); c = np.full(L, 0.5)
    T = []; Bq = []; V = []; W = []
    for t in range(gens):
        if psi > 0 and rng.random() < psi:
            j = rng.choice(L, size=n_flip, replace=False); theta[j] = 1 - theta[j]
        chat = a[:, None] * c[None, :] + (1 - a[:, None]) * 0.5
        pg = hp.D(chat, b[:, None])
        p_ok = np.where(theta[None, :] == 1, pg, 1 - pg)
        q = (pop == 2); nq = q.sum(axis=1)
        wrong = ((pop != 2) & (pop != theta[None, :])).any(axis=1)
        logp = np.where(q, np.log(np.clip(p_ok, 1e-12, 1)), 0.0).sum(axis=1)
        ps = np.where(wrong, 0.0, np.exp(logp))
        F = np.full(N, FMIN); F[(~wrong) & (nq == 0)] = FMAX
        idx = np.where((~wrong) & (nq > 0))[0]
        if idx.size:
            i = rng.geometric(np.clip(ps[idx], 1e-12, 1.0)); ok = i <= G
            ff = np.full(idx.size, FMIN); ff[ok] = 1.0 + 19.0 * (G - i[ok]) / G
            F[idx] = ff
        v = nq.var(); T.append(q.mean()); V.append(v); W.append((F > FMIN).sum())
        Bq.append(np.cov(F, nq)[0, 1] / v if v > 1e-9 else np.nan)
        win = np.where(F > FMIN)[0]
        if win.size:
            m = rng.choice(win, size=min(k, win.size), replace=True)
            phen = np.where(pop[m] == 2, theta[None, :], pop[m])
            c = np.where(rng.random(phen.shape) < f, phen, 1 - phen).mean(axis=0)
        else:
            c = np.full(L, 0.5)
        pr = F / F.sum(); pa = rng.choice(N, N, p=pr); pb = rng.choice(N, N, p=pr)
        cx = rng.integers(0, L, N); msk = np.arange(L)[None, :] < cx[:, None]
        pop = np.where(msk, pop[pa], pop[pb]).astype(np.int8)
    T = np.array(T); Bq = np.array(Bq); V = np.array(V); W = np.array(W)
    tail = slice(-100, None)
    ok = (V[BURN:] >= VMIN) & np.isfinite(Bq[BURN:])
    return dict(
        pq=float(T[-50:].mean()),
        bq_tail=float(np.nanmean(Bq[tail])) if np.isfinite(Bq[tail]).any() else float('nan'),
        var_tail=float(V[tail].mean()),
        bq_traj=float(Bq[BURN:][ok].mean()) if ok.any() else float('nan'),
        n_gen_traj=int(ok.sum()),
        eff_traj=float((Bq[BURN:][ok] * np.sqrt(V[BURN:][ok])).mean()) if ok.any() else float('nan'),
        nwin=float(W[-50:].mean()),
        alive=bool(W[-50:].mean() > 0.5))

def one(arg):
    a0, b0, psi, s = arg
    r = run(s, psi, a0, b0, P0)
    r.update(a=a0, b=b0, psi=psi, seed=s)
    return r

if __name__ == "__main__":
    jobs = [(a, b, p, s) for a in A for b in B for p in PSI for s in range(NS)]
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    json.dump(res, open(os.path.join("..", "hn_periph8.json"), "w"), indent=0)
    print("saved", len(res), "runs -> ../hn_periph8.json")

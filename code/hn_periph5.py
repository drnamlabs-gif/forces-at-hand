# -*- coding: utf-8 -*-
"""hn_periph5.py -- BANG 2x2 XAC NHAN (song song 12 tien trinh):
hoc xa hoi (a) x nhua tinh khoi tao (p0) doi voi SU SONG SOT
duoi muc tieu DI CHUYEN.

hn_periph4 he lo, tai psi>0:
  a=0.0  -> Winner=0 voi MOI p0        (khong hoc xa hoi => chet)
  a=0.8  -> Winner=0 khi p0=0.2
            Winner=646 khi p0=0.8      (can CA HAI)
Do o day P(song sot) tren TUNG seed, khong lay trung binh nhoe.
Them cot b=0.5 de tach vai tro co che TRUNG UONG (rap khuon).
"""
import json, os, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from hn_periph4 import run

NS, GENS = 12, 400

def one(arg):
    psi,a0,b0,p0,s = arg
    tr,bq,w = run(seed=s,p0=p0,a0=a0,b0=b0,psi=psi,gens=GENS)
    return (psi,a0,b0,p0,
            float(w[-50:].mean()), float(tr[-50:].mean()),
            float(np.nanmean(bq[-100:])) if np.isfinite(bq[-100:]).any() else float('nan'))

if __name__=="__main__":
    cells=[(psi,a0,b0,p0) for psi in (0.0,0.05,0.25)
           for (a0,b0) in ((0.0,0.0),(0.8,0.0),(0.8,0.5))
           for p0 in (0.2,0.8)]
    jobs=[c+(s,) for c in cells for s in range(NS)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res=list(ex.map(one,jobs,chunksize=1))
    out=[]
    print("BANG 2x2 (+b): P(song sot) duoi muc tieu DI CHUYEN,"
          f" {NS} seed, {GENS} the he\n")
    print(f"{'psi':>6}{'a':>5}{'b':>6}{'p0':>6}"
          f"{'P(song)':>10}{'p(?)|song':>11}{'beta_q':>10}{'Winner TB':>11}")
    prev=None
    for c in cells:
        r=[x for x in res if x[:4]==c]
        alive=[x[4]>0.5 for x in r]
        pq=[x[5] for x,a in zip(r,alive) if a]
        bq=[x[6] for x,a in zip(r,alive) if a and x[6]==x[6]]
        rec=dict(psi=c[0],a0=c[1],b0=c[2],p0=c[3],
                 P_alive=float(np.mean(alive)),
                 pq=float(np.mean(pq)) if pq else float('nan'),
                 bq=float(np.mean(bq)) if bq else float('nan'),
                 nwin=float(np.mean([x[4] for x in r])))
        out.append(rec)
        if prev is not None and c[0]!=prev: print()
        prev=c[0]
        pqs="       -- " if rec['pq']!=rec['pq'] else f"{rec['pq']:>11.4f}"
        bqs="       -- " if rec['bq']!=rec['bq'] else f"{rec['bq']:>10.4f}"
        print(f"{c[0]:>6.2f}{c[1]:>5.1f}{c[2]:>6.1f}{c[3]:>6.1f}"
              f"{rec['P_alive']:>10.2f}{pqs}{bqs}{rec['nwin']:>11.0f}")
    json.dump(out,open(os.path.join("..","hn_periph5.json"),"w"),
              indent=1,ensure_ascii=False)
    print("\nDONE")

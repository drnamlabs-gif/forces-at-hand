# -*- coding: utf-8 -*-
"""hn_periph7.py -- DOI CHUNG: quet psi khi KHONG co hoc xa hoi (a=0).
hn_periph6 cho beta_q doi dau tai psi* ~ 0.063 khi a=0.8,b=0.5.
Cau hoi doi chung: khong co hoc xa hoi thi co mien psi nao quan the
song du lau de nhua tinh duoc chon loc DUONG khong?
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from hn_periph6 import run

NS=12
def one(arg):
    psi,s=arg
    return (psi,)+run(s,psi,0.0,0.0,0.8)

if __name__=="__main__":
    PSI=(0.0,0.005,0.01,0.02,0.03,0.05)
    jobs=[(p,s) for p in PSI for s in range(NS)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res=list(ex.map(one,jobs,chunksize=1))
    print("DOI CHUNG a=0, b=0, p(?) khoi tao 0.8\n")
    print(f"{'psi':>7}{'P(song)':>9}{'p(?)':>9}{'beta_q':>10}"
          f"{'beta*sd':>10}{'Winner':>8}")
    out=[]
    for psi in PSI:
        r=[x for x in res if x[0]==psi]
        al=[x[4]>0.5 for x in r]; g=[x for x,a in zip(r,al) if a]
        if not g:
            print(f"{psi:>7.3f}{0.0:>9.2f}{'--':>9}{'--':>10}{'--':>10}{0:>8}")
            out.append(dict(psi=psi,P=0.0)); continue
        pq=np.mean([x[1] for x in g]); bq=np.nanmean([x[2] for x in g])
        vv=np.mean([x[3] for x in g]); w=np.mean([x[4] for x in g])
        out.append(dict(psi=psi,P=float(np.mean(al)),pq=float(pq),bq=float(bq),
                        var_nq=float(vv),eff=float(bq*np.sqrt(vv)),nwin=float(w)))
        print(f"{psi:>7.3f}{np.mean(al):>9.2f}{pq:>9.4f}{bq:>10.4f}"
              f"{bq*np.sqrt(vv):>+10.4f}{w:>8.0f}")
    json.dump(out,open(os.path.join("..","hn_periph7.json"),"w"),
              indent=1,ensure_ascii=False)
    print("\nDONE")

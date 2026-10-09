# -*- coding: utf-8 -*-
"""hn_periph6.py -- TIM DIEM DOI DAU cua beta_q theo psi.
hn_periph5: beta_q am o psi=0 va psi=0.05, DUONG o psi=0.25.
=> quet psi de dinh vi nguong. Bao ca Var(nq) de kiem tra beta_q
   khong phai nhieu khi p(?) -> 1 (nq don dieu, phuong sai nho).
"""
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import hn_periph as hp
from hn_periph3 import L,G,FMAX,FMIN

NS,GENS=12,400

def run(seed,psi,a0,b0,p0,N=1000,k=10,f=0.9,n_flip=1,gens=GENS):
    rng=np.random.default_rng(seed)
    theta=np.ones(L,dtype=np.int8)
    pop=rng.choice([0,1,2],size=(N,L),p=((1-p0)/2,(1-p0)/2,p0)).astype(np.int8)
    a=np.full(N,a0); b=np.full(N,b0); c=np.full(L,0.5)
    T=[];B=[];V=[];W=[]
    for t in range(gens):
        if psi>0 and rng.random()<psi:
            j=rng.choice(L,size=n_flip,replace=False); theta[j]=1-theta[j]
        chat=a[:,None]*c[None,:]+(1-a[:,None])*0.5
        pg=hp.D(chat,b[:,None])
        p_ok=np.where(theta[None,:]==1,pg,1-pg)
        q=(pop==2); nq=q.sum(axis=1)
        wrong=((pop!=2)&(pop!=theta[None,:])).any(axis=1)
        logp=np.where(q,np.log(np.clip(p_ok,1e-12,1)),0.0).sum(axis=1)
        ps=np.where(wrong,0.0,np.exp(logp))
        F=np.full(N,FMIN); F[(~wrong)&(nq==0)]=FMAX
        idx=np.where((~wrong)&(nq>0))[0]
        if idx.size:
            i=rng.geometric(np.clip(ps[idx],1e-12,1.0)); ok=i<=G
            ff=np.full(idx.size,FMIN); ff[ok]=1.0+19.0*(G-i[ok])/G; F[idx]=ff
        v=nq.var(); T.append(q.mean()); V.append(v); W.append((F>FMIN).sum())
        B.append(np.cov(F,nq)[0,1]/v if v>1e-9 else np.nan)
        win=np.where(F>FMIN)[0]
        if win.size:
            m=rng.choice(win,size=min(k,win.size),replace=True)
            phen=np.where(pop[m]==2,theta[None,:],pop[m])
            c=np.where(rng.random(phen.shape)<f,phen,1-phen).mean(axis=0)
        else: c=np.full(L,0.5)
        pr=F/F.sum(); pa=rng.choice(N,N,p=pr); pb=rng.choice(N,N,p=pr)
        cx=rng.integers(0,L,N); msk=np.arange(L)[None,:]<cx[:,None]
        pop=np.where(msk,pop[pa],pop[pb]).astype(np.int8)
    sl=slice(-100,None)
    return (float(np.mean(T[-50:])),float(np.nanmean(B[sl])),
            float(np.mean(V[sl])),float(np.mean(W[-50:])))

def one(arg): return arg[:2]+run(arg[2],arg[0],0.8,0.5,arg[1])

if __name__=="__main__":
    PSI=(0.0,0.02,0.05,0.08,0.12,0.18,0.25)
    jobs=[(psi,p0,s) for psi in PSI for p0 in (0.8,) for s in range(NS)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res=list(ex.map(one,jobs,chunksize=1))
    print("QUET psi, cau hinh a=0.8, b=0.5, p(?) khoi tao 0.8\n")
    print(f"{'psi':>6}{'P(song)':>9}{'p(?)':>9}{'beta_q':>10}"
          f"{'Var(nq)':>10}{'beta*sd':>10}{'Winner':>8}")
    out=[]
    for psi in PSI:
        r=[x for x in res if x[0]==psi]
        al=[x[5]>0.5 for x in r]
        g=[x for x,a in zip(r,al) if a]
        if not g:
            print(f"{psi:>6.2f}{0.0:>9.2f}{'--':>9}{'--':>10}{'--':>10}{'--':>10}{0:>8}")
            out.append(dict(psi=psi,P=0.0)); continue
        pq=np.mean([x[2] for x in g]); bq=np.nanmean([x[3] for x in g])
        vv=np.mean([x[4] for x in g]); w=np.mean([x[5] for x in g])
        # beta*sd = do lech fitness ky vong khi nq lech 1 SD -> dat trong thang F=1..20
        eff=bq*np.sqrt(vv)
        out.append(dict(psi=psi,P=float(np.mean(al)),pq=float(pq),bq=float(bq),
                        var_nq=float(vv),eff=float(eff),nwin=float(w)))
        print(f"{psi:>6.2f}{np.mean(al):>9.2f}{pq:>9.4f}{bq:>10.4f}"
              f"{vv:>10.3f}{eff:>+10.4f}{w:>8.0f}")
    json.dump(out,open(os.path.join("..","hn_periph6.json"),"w"),
              indent=1,ensure_ascii=False)
    print("\nDONE")

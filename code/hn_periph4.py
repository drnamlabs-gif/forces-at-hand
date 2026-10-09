# -*- coding: utf-8 -*-
"""hn_periph4.py -- MOI TRUONG DONG: plasticity co duoc CHON LOC DUONG khong?

hn_periph3 cho thay: moi truong TINH + social learning (a=1) => chon loc len '?'
bi TRUNG TINH HOA (p(?) nam yen tai gia tri khoi tao).
Cau hoi con lai: khi muc tieu DI CHUYEN (psi>0), '?' co duoc chon loc DUONG?
Phep thu: xuat phat p0=0.2 (thap). Neu p(?) TANG len thi la chon loc duong that.
"""
import json, os
import numpy as np
import hn_periph as hp
from hn_periph3 import L,G,FMAX,FMIN

def run(N=1000, gens=600, seed=0, k=10, f=0.9, p0=0.5,
        a0=0.8, b0=0.0, psi=0.0, n_flip=1):
    rng=np.random.default_rng(seed)
    theta=np.ones(L,dtype=np.int8)
    pop=rng.choice([0,1,2],size=(N,L),
                   p=((1-p0)/2,(1-p0)/2,p0)).astype(np.int8)
    a=np.full(N,a0); b=np.full(N,b0)
    c=np.full(L,0.5); traj=[]; bqs=[]; nw=[]
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
        F=np.full(N,FMIN); perf=(~wrong)&(nq==0); F[perf]=FMAX
        idx=np.where((~wrong)&(nq>0))[0]
        if idx.size:
            i=rng.geometric(np.clip(ps[idx],1e-12,1.0)); ok=i<=G
            ff=np.full(idx.size,FMIN); ff[ok]=1.0+19.0*(G-i[ok])/G; F[idx]=ff
        traj.append(float(q.mean())); nw.append(int((F>FMIN).sum()))
        v=nq.var(); bqs.append(float(np.cov(F,nq)[0,1]/v) if v>1e-9 else np.nan)
        win=np.where(F>FMIN)[0]
        if win.size:
            m=rng.choice(win,size=min(k,win.size),replace=True)
            phen=np.where(pop[m]==2,theta[None,:],pop[m])
            seen=np.where(rng.random(phen.shape)<f,phen,1-phen)
            c=seen.mean(axis=0)
        else: c=np.full(L,0.5)
        pr=F/F.sum(); pa=rng.choice(N,N,p=pr); pb=rng.choice(N,N,p=pr)
        cx=rng.integers(0,L,N); msk=np.arange(L)[None,:]<cx[:,None]
        pop=np.where(msk,pop[pa],pop[pb]).astype(np.int8)
    return np.array(traj),np.array(bqs),np.array(nw)

if __name__=="__main__":
    NS=12; out=[]
    print("MOI TRUONG DONG: p0=0.2 (thap) -> p(?) co TANG khong?\n")
    print(f"{'a':>5}{'psi':>7}{'p0':>6}{'p(?) cuoi':>11}{'thay doi':>10}"
          f"{'beta_q':>10}{'Winner':>8}")
    for a0 in (0.0,0.8):
        for psi in (0.0,0.05,0.25):
            for p0 in (0.2,0.8):
                R=[run(seed=s,p0=p0,a0=a0,psi=psi) for s in range(NS)]
                end=float(np.mean([r[0][-50:].mean() for r in R]))
                bq =float(np.nanmean([np.nanmean(r[1][-100:]) for r in R]))
                w  =float(np.mean([r[2][-50:].mean() for r in R]))
                out.append(dict(a0=a0,psi=psi,p0=p0,end=end,
                                delta=end-p0,bq=bq,nwin=w))
                print(f"{a0:>5.1f}{psi:>7.2f}{p0:>6.1f}{end:>11.4f}"
                      f"{end-p0:>+10.4f}{bq:>10.4f}{w:>8.0f}")
            lo=[o for o in out if o['a0']==a0 and o['psi']==psi]
            sp=abs(lo[0]['end']-lo[1]['end'])
            print(f"{'':>18}spread cuoi = {sp:.4f} / spread dau 0.60"
                  f" = {sp/0.6*100:.1f}% con nho khoi tao\n")
    json.dump(out,open(os.path.join("..","hn_periph4.json"),"w"),
              indent=1,ensure_ascii=False)
    print("DONE")

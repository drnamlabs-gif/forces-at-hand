# -*- coding: utf-8 -*-
"""hn_periph3.py -- PHEP THU PHAN DINH: p(?) ~ 0.5 la CHON LOC hay TRUNG TINH?

Neu CHON LOC (co diem can bang thuc su): xuat phat tu p0 = 0.2 / 0.5 / 0.8
   deu HOI TU ve cung mot gia tri.
Neu TRUNG TINH (s -> 0, chi la drift quanh gia tri khoi tao):
   moi p0 NAM YEN tai cho -> hoi tu KHONG xay ra.

Doi chieu voi ket qua Morgan (5.7): van hoa TRUNG TINH HOA chon loc,
s -> 0 => x -> 1/2. Neu H&N cung vay thi hai ket qua ung ho nhau.
"""
import json, os
import numpy as np
import hn_periph as hp

L,G,FMAX,FMIN = 20,1000,20.0,1.0

def run(N=1000, gens=600, seed=0, k=10, f=0.9, p0=0.5,
        a0=0.0, b0=0.0, mut=0.0):
    rng=np.random.default_rng(seed)
    theta=np.ones(L,dtype=np.int8)
    pop=rng.choice([0,1,2],size=(N,L),
                   p=((1-p0)/2,(1-p0)/2,p0)).astype(np.int8)
    a=np.full(N,a0); b=np.full(N,b0)
    c=np.full(L,0.5); traj=[]
    for t in range(gens):
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
        traj.append(float(q.mean()))
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
        if mut>0:
            fl=rng.random(pop.shape)<mut
            pop=np.where(fl,rng.choice([0,1,2],size=pop.shape),pop).astype(np.int8)
    return np.array(traj)

if __name__=="__main__":
    NS=12; out=[]
    print("PHEP THU HOI TU: p(?) khoi tao khac nhau -> cuoi cung ve dau?\n")
    print(f"{'che do':>22}{'p0':>6}{'p(?) g100':>11}{'p(?) g300':>11}"
          f"{'p(?) cuoi':>11}{'|cuoi-p0|':>11}")
    for lab,a0 in (("H&N goc a=0",0.0),("a=0.8",0.8),("a=1.0",1.0)):
        row=[]
        for p0 in (0.2,0.5,0.8):
            T=np.array([run(seed=s,p0=p0,a0=a0) for s in range(NS)])
            m=T.mean(axis=0)
            r=dict(lab=lab,a0=a0,p0=p0,g100=float(m[100]),g300=float(m[300]),
                   end=float(m[-50:].mean()))
            r["drift"]=abs(r["end"]-p0); out.append(r); row.append(r)
            print(f"{lab:>22}{p0:>6.1f}{r['g100']:>11.4f}{r['g300']:>11.4f}"
                  f"{r['end']:>11.4f}{r['drift']:>11.4f}")
        sp=max(x["end"] for x in row)-min(x["end"] for x in row)
        verdict=("HOI TU -> co diem can bang, CHON LOC that"
                 if sp<0.05 else
                 "KHONG hoi tu -> phu thuoc khoi tao, TRUNG TINH/gan trung tinh")
        print(f"{'':>22}  spread(cuoi) = {sp:.4f}  -> {verdict}\n")
    json.dump(out,open(os.path.join("..","hn_periph3.json"),"w"),
              indent=1,ensure_ascii=False)
    print("DONE")

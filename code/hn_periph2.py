# -*- coding: utf-8 -*-
"""hn_periph2.py -- (1) chay lai voi tham so quan the SONG DUOC
                    (2) plasticity duoc CHON LOC hay chi duoc DUNG THA?

Do truc tiep GRADIENT CHON LOC len so dau '?':
     beta_q = Cov(F, q) / Var(q)   trong tung the he
  beta_q > 0 : plasticity duoc UNG HO (instructive)
  beta_q ~ 0 : plasticity TRUNG TINH, chi khong bi loai (permissive)
  beta_q < 0 : plasticity bi loai
"""
import json, os
import numpy as np
import hn_periph as hp

L,G,FMAX,FMIN = 20,1000,20.0,1.0

def run(N=1000, gens=400, seed=0, psi=0.0, n_flip=1, k=10, f=0.9,
        mut_ab=0.03, evolve=True, a0=0.0, b0=0.0):
    rng=np.random.default_rng(seed)
    theta=np.ones(L,dtype=np.int8)
    pop=rng.choice([0,1,2],size=(N,L),p=(0.25,0.25,0.5)).astype(np.int8)
    a=rng.uniform(0,1,N) if evolve else np.full(N,a0)
    b=rng.uniform(-1,1,N) if evolve else np.full(N,b0)
    c=np.full(L,0.5)
    H={key:[] for key in ("a","b","pq","nwin","bq")}
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
        # GRADIENT CHON LOC len so dau '?'
        vq=nq.var()
        bq=float(np.cov(F,nq)[0,1]/vq) if vq>1e-9 else np.nan
        for key,v in (("a",a.mean()),("b",b.mean()),("pq",q.mean()),
                      ("nwin",(F>FMIN).sum()),("bq",bq)):
            H[key].append(float(v))
        win=np.where(F>FMIN)[0]
        if win.size:
            m=rng.choice(win,size=min(k,win.size),replace=True)
            phen=np.where(pop[m]==2,theta[None,:],pop[m])
            seen=np.where(rng.random(phen.shape)<f,phen,1-phen)
            c=seen.mean(axis=0)
        else: c=np.full(L,0.5)
        pr=F/F.sum(); pa=rng.choice(N,N,p=pr); pb=rng.choice(N,N,p=pr)
        cx=rng.integers(0,L,N); msk=np.arange(L)[None,:]<cx[:,None]
        newpop=np.where(msk,pop[pa],pop[pb])
        ac=np.where(rng.random(N)<0.5,a[pa],a[pb]); bc=np.where(rng.random(N)<0.5,b[pa],b[pb])
        if evolve:
            ac=np.clip(ac+rng.normal(0,mut_ab,N),0,1)
            bc=np.clip(bc+rng.normal(0,mut_ab,N),-1,1)
        pop,a,b=newpop.astype(np.int8),ac,bc
    return {key:np.array(v) for key,v in H.items()}

def tl(h,key,fr=0.2):
    x=h[key]; x=x[-int(len(x)*fr):]
    return float(np.nanmean(x))

if __name__=="__main__":
    NS=12
    print("A. CHAY LAI voi n_flip=1 (quan the song duoc)\n")
    print(f"{'psi':>6}{'a TB':>9}{'a SD':>8}{'b TB':>9}{'b SD':>8}"
          f"{'p(?)':>8}{'Winner':>8}")
    out=[]
    for psi in (0.0,0.02,0.05,0.10,0.25):
        A=[run(seed=s,psi=psi) for s in range(NS)]
        av=np.array([tl(h,"a") for h in A]); bv=np.array([tl(h,"b") for h in A])
        r=dict(psi=psi,a=float(av.mean()),a_sd=float(av.std()),
               b=float(bv.mean()),b_sd=float(bv.std()),
               pq=float(np.mean([tl(h,"pq") for h in A])),
               nwin=float(np.mean([tl(h,"nwin") for h in A])))
        out.append(r)
        print(f"{psi:>6.2f}{r['a']:>9.4f}{r['a_sd']:>8.4f}{r['b']:>9.4f}"
              f"{r['b_sd']:>8.4f}{r['pq']:>8.4f}{r['nwin']:>8.0f}")
    A_=np.array([o["a_sd"] for o in out]); B_=np.array([o["b_sd"] for o in out])
    print(f"\n  SD giua cac lan chay: a={A_.mean():.4f}  b={B_.mean():.4f}"
          f"  -> ty le {B_.mean()/A_.mean():.2f}x")

    print("\n\nB. PLASTICITY duoc CHON LOC hay chi TRUNG TINH?")
    print("   beta_q = Cov(F, so dau '?') / Var(so dau '?')\n")
    print(f"{'che do':>24}{'beta_q':>11}{'p(?) cuoi':>11}{'Winner':>8}")
    out2=[]
    for lab,kw in (("H&N goc, a=0",dict(evolve=False,a0=0.0,b0=0.0)),
                   ("a=0.8 co dinh, b=0",dict(evolve=False,a0=0.8,b0=0.0)),
                   ("a=1.0 co dinh, b=0",dict(evolve=False,a0=1.0,b0=0.0)),
                   ("a,b TIEN HOA",dict(evolve=True))):
        A=[run(seed=s,**kw) for s in range(NS)]
        r=dict(lab=lab,bq=float(np.mean([tl(h,"bq") for h in A])),
               pq=float(np.mean([tl(h,"pq") for h in A])),
               nwin=float(np.mean([tl(h,"nwin") for h in A])))
        out2.append(r)
        print(f"{lab:>24}{r['bq']:>11.4f}{r['pq']:>11.4f}{r['nwin']:>8.0f}")
    json.dump(dict(psi=out,grad=out2),
              open(os.path.join("..","hn_periph2.json"),"w"),indent=1,ensure_ascii=False)
    print("\nDONE")

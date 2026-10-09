# -*- coding: utf-8 -*-
"""hn_periph.py -- kiem dinh NGOAI VI so voi TRUNG TAM (Heyes, Chater & Dwyer 2020)
tren landscape H&N.

VAN DE cua ban truoc (hn_w.py): chi co MOT tham so w, gop ca ngoai vi lan
trung tam, nen khong the kiem duoc bat doi xung ma bai cua ho khang dinh.

TACH HAI THAM SO, ca hai deu nam trong kieu gen:
   a  NGOAI VI (chu y): bao nhieu phan tin hieu xa hoi thuc su di vao
        c_hat_j = a * c_j + (1-a) * 0.5
   b  TRUNG TAM (chien luoc): phep bien doi tin hieu thanh phan doan
        p_j = D_b(c_hat_j) = c + b*c*(1-c)*(2c-1)      (Boyd & Richerson)
        b>0 tuan thu (khuech dai da so); b=0 tuyen tinh; b<0 nghich tuan thu

DIEU KIEN TRUNG PHAT CHUYEN BIET: psi = xac suat MOI THE HE dich doi o mot
so locus. Khi dich doi, da so phan anh dich CU. Nen:
   - a van co ich (thong tin cu con mot phan gia tri)
   - b>0 khoa vao qua khu  -> toi uu cua b PHU THUOC psi
Du doan cua Heyes 2020 dich sang day: chon loc NAM CHAC a, MAT DAN b.
Do luong: ca gia tri trung binh VA phuong sai giua cac lan chay.
"""
import json, os
import numpy as np

L, G, FMAX, FMIN = 20, 1000, 20.0, 1.0

def D(c, b):
    return np.clip(c + b*c*(1-c)*(2*c-1), 1e-9, 1-1e-9)

def run(N=1000, gens=400, seed=0, psi=0.0, n_flip=3, k=10, f=0.9,
        mut=0.0, mut_ab=0.03, evolve=True, a0=None, b0=None):
    rng=np.random.default_rng(seed)
    theta=np.ones(L,dtype=np.int8)                    # dich, co the doi
    pop=rng.choice([0,1,2],size=(N,L),p=(0.25,0.25,0.5)).astype(np.int8)
    a=rng.uniform(0,1,N) if evolve else np.full(N,a0)
    b=rng.uniform(-1,1,N) if evolve else np.full(N,b0)
    c=np.full(L,0.5)
    H={key:[] for key in ("a","b","pq","nwin")}
    for t in range(gens):
        if psi>0 and rng.random()<psi:                # dich doi o n_flip locus
            j=rng.choice(L,size=n_flip,replace=False); theta[j]=1-theta[j]
        chat=a[:,None]*c[None,:]+(1-a[:,None])*0.5
        p_guess=D(chat,b[:,None])
        # xac suat doan DUNG tai locus j = p neu theta_j=1, nguoc lai 1-p
        p_ok=np.where(theta[None,:]==1, p_guess, 1-p_guess)
        q=(pop==2); wrong=((pop!=2)&(pop!=theta[None,:])).any(axis=1)
        logp=np.where(q,np.log(np.clip(p_ok,1e-12,1)),0.0).sum(axis=1)
        psucc=np.where(wrong,0.0,np.exp(logp))
        F=np.full(N,FMIN); perf=(~wrong)&(~q.any(axis=1)); F[perf]=FMAX
        idx=np.where((~wrong)&q.any(axis=1))[0]
        if idx.size:
            i=rng.geometric(np.clip(psucc[idx],1e-12,1.0)); ok=i<=G
            ff=np.full(idx.size,FMIN); ff[ok]=1.0+19.0*(G-i[ok])/G; F[idx]=ff
        for key,v in (("a",a.mean()),("b",b.mean()),("pq",q.mean()),
                      ("nwin",(F>FMIN).sum())):
            H[key].append(float(v))
        # cue xa hoi tu k ca the thanh cong
        win=np.where(F>FMIN)[0]
        if win.size:
            m=rng.choice(win,size=min(k,win.size),replace=True)
            phen=np.where(pop[m]==2, theta[None,:], pop[m])
            seen=np.where(rng.random(phen.shape)<f, phen, 1-phen)
            c=seen.mean(axis=0)
        else: c=np.full(L,0.5)
        pr=F/F.sum(); pa=rng.choice(N,N,p=pr); pb=rng.choice(N,N,p=pr)
        cx=rng.integers(0,L,N); msk=np.arange(L)[None,:]<cx[:,None]
        newpop=np.where(msk,pop[pa],pop[pb])
        ac=np.where(rng.random(N)<0.5,a[pa],a[pb])
        bc=np.where(rng.random(N)<0.5,b[pa],b[pb])
        if evolve:
            ac=np.clip(ac+rng.normal(0,mut_ab,N),0,1)
            bc=np.clip(bc+rng.normal(0,mut_ab,N),-1,1)
        if mut>0:
            mm=rng.random((N,L))<mut
            if mm.any(): newpop[mm]=rng.integers(0,3,size=int(mm.sum()))
        pop,a,b=newpop.astype(np.int8),ac,bc
    return {key:np.array(v) for key,v in H.items()}

def tail(h,key,fr=0.2):
    x=h[key]; return float(x[-int(len(x)*fr):].mean())

if __name__=="__main__":
    print("KIEM DINH NGOAI VI so voi TRUNG TAM tren H&N")
    print("  a = chu y (ngoai vi)   b = tuan thu (trung tam)")
    print("  psi = xac suat dich doi moi the he\n")
    print(f"{'psi':>6}{'a TB':>9}{'a SD':>9}{'b TB':>10}{'b SD':>9}"
          f"{'p(?)':>9}{'Winner':>9}")
    NS=12; out=[]
    for psi in (0.0,0.02,0.05,0.10,0.25):
        A=[run(seed=s,psi=psi) for s in range(NS)]
        av=np.array([tail(h,"a") for h in A]); bv=np.array([tail(h,"b") for h in A])
        r=dict(psi=psi,a=float(av.mean()),a_sd=float(av.std()),
               b=float(bv.mean()),b_sd=float(bv.std()),
               pq=float(np.mean([tail(h,"pq") for h in A])),
               nwin=float(np.mean([tail(h,"nwin") for h in A])))
        out.append(r)
        print(f"{psi:>6.2f}{r['a']:>9.4f}{r['a_sd']:>9.4f}{r['b']:>10.4f}"
              f"{r['b_sd']:>9.4f}{r['pq']:>9.4f}{r['nwin']:>9.0f}")
    A=np.array([o["a_sd"] for o in out]); B=np.array([o["b_sd"] for o in out])
    print(f"\n  phuong sai giua cac lan chay, trung binh tren psi:")
    print(f"    a (ngoai vi): SD = {A.mean():.4f}")
    print(f"    b (trung tam): SD = {B.mean():.4f}   -> ty le {B.mean()/A.mean():.1f}x")
    print(f"  bien thien cua TRUNG BINH tren dai psi:")
    print(f"    a: {max(o['a'] for o in out)-min(o['a'] for o in out):.4f}")
    print(f"    b: {max(o['b'] for o in out)-min(o['b'] for o in out):.4f}")
    json.dump(out,open(os.path.join("..","hn_periph.json"),"w"),indent=1)
    print("\nDONE")


# ====================================================================
# GHI CHU VE TINH GIAI TICH DUOC
# Thiet ke nay GIU NGUYEN xuong song cua Belew, tung buoc mot:
#
#   (i)   moi locus '?' la mot BERNOULLI doc lap, xac suat doan dung p_ok_j
#   (ii)  mot lan thu thanh cong khi MOI locus dung:  P = prod_j p_ok_j
#         => log P = sum_j log p_ok_j          (CONG TINH tren locus)
#   (iii) so lan thu toi thanh cong ~ HINH HOC(P), cat cut tai G
#         => E(g) = (1 - (1-P)^G) / P          [Belew, pt (3.5)]
#   (iv)  fitness F = 20 - 19*E(g)/G           [Belew, pt (3.3)]
#
# Nen moi cong thuc cua chuong 3 va 4 ap thang duoc. Gradient chon loc
# len hai tham so la dao ham so cap:
#
#   dP/da  = P * sum_j (1/p_j) * D'(chat_j) * (c_j - 1/2)
#           voi D(c) = c + b(-2c^3 + 3c^2 - c),  D'(c) = 1 + b(-6c^2 + 6c - 1)
#   dP/db  = P * sum_j (1/p_j) * chat_j(1-chat_j)(2*chat_j-1)
#   dF/dx  = -19/G * dE(g)/dP * dP/dx,
#            dE(g)/dP = [G(1-P)^(G-1) P - (1-(1-P)^G)] / P^2
#
# Phi tuyen duy nhat la D_b, mot da thuc bac ba. Khong co ham dac biet nao.
# ====================================================================

def grad_check(a=0.7,b=0.3,q=8,cbar=0.85,G=1000):
    """Kiem dao ham dang dong so voi sai phan huu han."""
    import numpy as np
    def P(a,b):
        chat=a*cbar+(1-a)*0.5
        p=D(np.array([chat]),np.array([b]))[0]
        return p**q
    def Eg(P): return (1-(1-P)**G)/P
    def F(a,b): return 20-19*Eg(P(a,b))/G
    h=1e-6
    dFa=(F(a+h,b)-F(a-h,b))/(2*h); dFb=(F(a,b+h)-F(a,b-h))/(2*h)
    chat=a*cbar+(1-a)*0.5
    p=float(D(np.array([chat]),np.array([b]))[0]); Pv=p**q
    Dp=1+b*(-6*chat**2+6*chat-1)                    # D'(c), dau da sua
    dPa=Pv*q*(1/p)*Dp*(cbar-0.5)
    dPb=Pv*q*(1/p)*chat*(1-chat)*(2*chat-1)
    dEdP=(G*(1-Pv)**(G-1)*Pv-(1-(1-Pv)**G))/Pv**2
    k=-19.0/G*dEdP
    print("Kiem gradient dang dong so voi sai phan huu han:")
    print(f"   dF/da: dang dong {k*dPa:+.6e}   sai phan {dFa:+.6e}")
    print(f"   dF/db: dang dong {k*dPb:+.6e}   sai phan {dFb:+.6e}")

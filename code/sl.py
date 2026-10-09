# -*- coding: utf-8 -*-
"""
sl.py -- Mo hinh SOCIAL LEARNING thong nhat tren landscape Hinton & Nowlan.

Nguyen tac thiet ke (Heyes 2011, "What's social about social learning?"):
    Chi co MOT co che hoc. Hoc xa hoi va hoc ca the dung CUNG mot co che;
    chung chi khac nhau o DAU VAO. Cai duoc chuyen hoa thich nghi cho doi song
    xa hoi khong phai "learning mechanism" ma la "INPUT MECHANISM" -- cac qua
    trinh tri giac / chu y / dong co huong ve dong loai.

Vi vay o day KHONG co thu tuc "imitation" rieng. Moi ca the chay cung mot
vong hoc; thong tin xa hoi chi lam LECH phan phoi lay mau tai cac locus '?':

    theta_i = (1 - w) * 0.5  +  w * D_gamma(c_i)

    w      trong so xa hoi   -- input mechanism cua Heyes; CO THE TIEN HOA
    c_i    cue xa hoi        -- ti le '1' trong k gia tri CAM NHAN DUOC
    f      fidelity          -- moi gia tri cam nhan dung voi xac suat f
    gamma  conformity        -- D(p) = p + gamma*p(1-p)(2p-1)  (Boyd&Richerson)

Chi phi hoc GIONG NHAU cho moi ca the: toi da G lan thu, F = 1 + 19(G-n)/G.
Khong giam gia rieng cho hoc xa hoi. Loi ich xa hoi den tu viec moi lan thu
co xac suat trung cao hon -- khong phai tu viec duoc mien phi.

Khong co "oracle": nguoi hoc KHONG kiem tra thay co dung khong truoc khi
chiu anh huong. Nho vay mo hinh co the bieu hien nghich ly Rogers.
"""
import numpy as np

INCORRECT, CORRECT, UNDECIDED = 0, 1, 2
FMIN, FMAX = 1.0, 20.0


# ============================================================ CUE XA HOI
def conformity(p, gamma):
    """D(p) = p + gamma*p(1-p)(2p-1).  Boyd & Richerson (1985).
    gamma > 0 : tuan thu (khuech dai da so)   gamma < 0 : nghich tuan thu
    gamma = 0 : tuyen tinh (khong thien lech tan so)"""
    if gamma == 0.0:
        return p
    return np.clip(p + gamma * p * (1 - p) * (2 * p - 1), 0.0, 1.0)


def social_cue(models_pheno, f, gamma, rng):
    """models_pheno : (N, k, L) gia tri bieu hien cua k mau ma moi ca the quan sat.
    Tra ve c : (N, L) in [0,1].

    Fidelity ap o muc TRI GIAC tung locus tung mau: gia tri cam nhan dung voi
    xac suat f, nguoc lai voi 1-f. Day dung la co che fidelity cua Le (2019),
    nhung dat vao dau vao chu khong vao mot thu tuc copy rieng.
    """
    ok = rng.random(models_pheno.shape) < f
    perceived = np.where(ok, models_pheno, 1 - models_pheno)
    p = perceived.mean(axis=1)                    # (N, L) ti le '1' trong k mau
    return conformity(p, gamma)


def theta_from_cue(c, w):
    """theta = (1-w)*0.5 + w*c.  w = 0 -> hoc ca the thuan tuy (H&N)."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    return (1.0 - w) * 0.5 + w * c


# ============================================================ HOC + FITNESS
def learn(geno, theta, target, G, rng, obs_cost=0.0, k=1, budget=None):
    """Mot vong doi hoc, giong het nhau cho moi ca the.

    Moi lan thu: moi locus '?' lay mau Bernoulli(theta_i).
    Xac suat trung TRONG MOT LAN THU:
        P = prod_{i in '?'}  [theta_i neu target_i=1 else 1-theta_i]
    (bang 0 neu genotype co allele co dinh SAI).
    So lan thu toi khi trung ~ Hinh hoc(P) cat cut tai G  -- lay mau truc tiep,
    tuong duong CHINH XAC voi viec mo phong tung lan thu.

    obs_cost : chi phi quan sat, tinh bang so 'lan thu' cho moi locus '?' moi mau
               (Heyes: input mechanism cung ton tai nguyen). Mac dinh 0.
    budget   : SO LAN THU THUC SU DUOC PHEP (mac dinh = G).
               Tach roi khoi G, vi G la THANG GIA fitness (moi lan thu ton
               19/G diem) con budget la NGAN SACH tim kiem. Tach hai thu nay
               moi ablate duoc "co bao nhieu tim kiem ca the" ma khong lam
               thay doi thang do fitness.
    """
    if budget is None:
        budget = G
    N, L = geno.shape
    isq = geno == UNDECIDED
    q = isq.sum(axis=1)

    match = np.where(target[None, :] == 1, theta, 1.0 - theta)
    with np.errstate(divide="ignore"):
        logP = np.where(isq, np.log(np.clip(match, 1e-300, 1.0)), 0.0).sum(axis=1)
    P = np.exp(logP)

    wrong_fixed = ((geno != UNDECIDED) & (geno != target[None, :])).any(axis=1)
    P = np.where(wrong_fixed, 0.0, P)

    n = np.full(N, G + 1, dtype=float)
    live = P > 0
    if live.any():
        n[live] = rng.geometric(P[live])
    n = n + obs_cost * k * q                      # cong chi phi quan sat
    success = n <= budget

    F = np.where(success, 1.0 + (FMAX - 1.0) * (G - n) / G, FMIN)
    F = np.maximum(F, FMIN)

    # kieu hinh bieu hien: thanh cong -> dung target; that bai -> lan thu cuoi
    pheno = np.where(isq, (rng.random((N, L)) < theta).astype(np.int8), geno)
    pheno[success] = np.where(isq[success], target[None, :], geno[success])
    return F, pheno.astype(np.int8), q, success


def learn_switch(geno, prev_pheno, prev_F, model_idx, target, G, rng,
                 fidelity=1.0, cost="q"):
    """Cai lai DUNG Algorithm 3 cua Le (2019), de doi chieu.

        neu demonstrator DUNG (fitness > 1):  chep het cac locus '?' cua no,
                                              chi phi = q lan thu  (cost="q")
                                              hoac 1 lan thu       (cost="trial")
        nguoc lai:                            tu hoc ngau nhien, toi da G lan

    Day la mot CONG TAC giua hai thu tuc, khong phai mot trong so.
    Co "oracle": nguoi hoc biet mien phi thay dung hay sai.
    """
    N, L = geno.shape
    isq = geno == UNDECIDED
    q = isq.sum(axis=1)
    wrong_fixed = ((geno != UNDECIDED) & (geno != target[None, :])).any(axis=1)

    demon_ok = prev_F[model_idx] > FMIN            # <- cong oracle
    n = np.full(N, G + 1, dtype=float)

    # --- nhanh 1: bat chuoc mot lan ---
    sl_ = demon_ok & ~wrong_fixed
    if sl_.any():
        v = prev_pheno[model_idx][sl_]
        ok = rng.random(v.shape) < fidelity
        got = np.where(ok, v, 1 - v)
        hit = np.all(np.where(isq[sl_], got == target[None, :], True), axis=1)
        c = q[sl_].astype(float) if cost == "q" else np.ones(sl_.sum())
        nn = np.where(hit, c, G + 1)
        n[np.where(sl_)[0]] = nn

    # --- nhanh 2: tu hoc ngau nhien ---
    il_ = ~demon_ok & ~wrong_fixed
    if il_.any():
        P = 0.5 ** q[il_]
        n[np.where(il_)[0]] = rng.geometric(P)

    success = n <= G
    F = np.where(success, 1.0 + (FMAX - 1.0) * (G - n) / G, FMIN)
    F = np.maximum(F, FMIN)
    pheno = np.where(isq, (rng.random((N, L)) < 0.5).astype(np.int8), geno)
    pheno[success] = np.where(isq[success], target[None, :], geno[success])
    return F, pheno.astype(np.int8), q, success


# ============================================================ CHON MAU (who)
def pick_models(F_prev, N, k, who, rng, parents=None):
    """Tra ve chi so (N, k) cua cac mau ma moi ca the quan sat."""
    M = len(F_prev)
    if who == "best":                    # mot demonstrator toan cuc (Le, thesis)
        return np.full((N, k), int(np.argmax(F_prev)))
    if who == "parent":                  # doc: bo/me tot hon (Le, ISAL 2019)
        if parents is None:              # the he 0 chua co bo me -> ngau nhien
            return rng.integers(0, M, size=(N, k))
        a, b = parents
        best = np.where(F_prev[a] >= F_prev[b], a, b)
        return np.repeat(best[:, None], k, axis=1)
    if who == "random":                  # khong thien lech
        return rng.integers(0, M, size=(N, k))
    if who == "payoff":                  # ti le fitness (Best 1999)
        p = F_prev / F_prev.sum()
        return rng.choice(M, size=(N, k), p=p)
    if who == "top_decile":              # ngau nhien trong 10% tot nhat
        top = np.argsort(F_prev)[-max(1, M // 10):]
        return top[rng.integers(0, len(top), size=(N, k))]
    raise ValueError(who)


# ============================================================ VONG TIEN HOA
def run(generations=200, N=1000, L=20, G=300, seed=0,
        w=0.0, k=1, fidelity=1.0, gamma=0.0, who="parent",
        mode="blend", switch_cost="q", plastic=True, budget=None,
        obs_cost=0.0, evolve_w=False, w_mut=0.02,
        target=None, change_freq=None, change_mag=0.1,
        mutation=0.0):
    """w co the la scalar, hoac evolve_w=True de w tro thanh mot gene tien hoa
    (thao tac hoa truc tiep luan diem cua Heyes: input mechanism la noi chiu
    chuyen hoa thich nghi).

    change_freq : neu dat, moi change_freq the he muc tieu doi (Dynamic NIH).
    """
    rng = np.random.default_rng(seed)
    if plastic:
        geno = rng.choice([INCORRECT, CORRECT, UNDECIDED], size=(N, L),
                          p=(0.25, 0.25, 0.50)).astype(np.int8)
    else:                                   # doi chung: TIEN HOA DON THUAN
        geno = rng.choice([INCORRECT, CORRECT], size=(N, L),
                          p=(0.5, 0.5)).astype(np.int8)
    if target is None:
        target = np.ones(L, dtype=np.int8)
    target = target.copy()

    wv = (rng.random(N) if evolve_w else np.full(N, float(w)))
    prev_pheno = rng.integers(0, 2, size=(N, L)).astype(np.int8)
    prev_F = np.ones(N)
    parents = None

    H = {key: [] for key in ("p1", "pq", "p0", "fmean", "fmax",
                             "n_winner", "w_mean", "cue_acc",
                             "wF_corr", "F_hiw", "F_low")}
    loci_q = np.zeros((generations, L))

    for t in range(generations):
        if change_freq and t > 0 and t % change_freq == 0:
            flip = rng.random(L) < change_mag
            target = np.where(flip, 1 - target, target).astype(np.int8)

        idx = pick_models(prev_F, N, k, who, rng, parents)
        models = prev_pheno[idx]                     # (N, k, L)
        c = social_cue(models, fidelity, gamma, rng)
        theta = theta_from_cue(c, wv)

        if mode == "blend":
            F, pheno, q, success = learn(geno, theta, target, G, rng,
                                         obs_cost, k, budget)
        else:
            F, pheno, q, success = learn_switch(
                geno, prev_pheno, prev_F, idx[:, 0], target, G, rng,
                fidelity, switch_cost)

        H["p1"].append(float((geno == CORRECT).mean()))
        H["pq"].append(float((geno == UNDECIDED).mean()))
        H["p0"].append(float((geno == INCORRECT).mean()))
        H["fmean"].append(float(F.mean())); H["fmax"].append(float(F.max()))
        H["n_winner"].append(int(success.sum()))
        H["w_mean"].append(float(wv.mean()))
        H["cue_acc"].append(float(np.where(target[None, :] == 1, c, 1 - c).mean()))
        # --- phep thu Rogers: w cao co loi CA THE khong? ---
        if wv.std() > 1e-9:
            H["wF_corr"].append(float(np.corrcoef(wv, F)[0, 1]))
            lo, hi = np.quantile(wv, [0.25, 0.75])
            H["F_hiw"].append(float(F[wv >= hi].mean()))
            H["F_low"].append(float(F[wv <= lo].mean()))
        else:
            for kk in ("wF_corr", "F_hiw", "F_low"): H[kk].append(np.nan)
        loci_q[t] = (geno == UNDECIDED).mean(axis=0)

        # --- sinh san ---
        p = F / F.sum()
        pa = rng.choice(N, size=N, p=p); pb = rng.choice(N, size=N, p=p)
        cut = rng.integers(1, L, size=N)
        mask = np.arange(L)[None, :] < cut[:, None]
        child = np.where(mask, geno[pa], geno[pb]).astype(np.int8)
        if mutation > 0:
            m = rng.random((N, L)) < mutation
            if m.any():
                child[m] = rng.integers(0, 3, size=int(m.sum()))
        if evolve_w:
            wc = np.where(rng.random(N) < 0.5, wv[pa], wv[pb])
            wv = np.clip(wc + rng.normal(0, w_mut, N), 0.0, 1.0)

        prev_pheno, prev_F, parents, geno = pheno, F, (pa, pb), child

    out = {key: np.asarray(v) for key, v in H.items()}
    out["loci_q"] = loci_q; out["geno"] = geno; out["target"] = target
    return out

# -*- coding: utf-8 -*-
"""Population '?' frequency for the long runs (English tables), same seeds."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sl

LONG = [("IL", dict(w=0.0, who='parent')),
        ("switch", dict(mode='switch', who='parent')),
        ("blend evo", dict(evolve_w=True, k=5, who='top_decile')),
        ("blend .8", dict(w=0.8, k=5, who='top_decile'))]

def one(a):
    i, s = a
    r = sl.run(generations=2500, seed=9000 + s, **LONG[i][1])
    return (i, float(r["pq"][-1]))

def drift(s):
    r = sl.run(generations=3000, seed=6000 + s, evolve_w=True,
               who="top_decile", k=5)
    return float(r["pq"][-1])

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, [(i, s) for i in range(4) for s in range(6)]))
        dq = list(ex.map(drift, range(6)))
    for i in range(4):
        print(LONG[i][0], round(float(np.mean([v for j, v in res if j == i])), 3))
    print("drift pq:", round(float(np.mean(dq)), 3))

# The forces at hand: social learning is permissive, not instructive, for the evolution of phenotypic plasticity

Code, simulation outputs and figures for the manuscript (Nam Le, submitted to
*Journal of the Royal Society Interface*).

The model is the Hinton–Nowlan landscape with a social channel split into a
heritable attention gate `a` and a heritable conformity transform `b`
(notation `w`, `gamma` in some older scripts). Selection on plasticity is
measured directly as the within-generation regression of fitness on the
number of undetermined loci.

## Layout

| path | contents |
|---|---|
| `code/` | all simulation and figure scripts (Python 3, NumPy, Matplotlib) |
| `results/` | archived JSON outputs used for every number in the paper |
| `figures/` | archived figures exactly as included in the paper |
| `figs/` | (empty) scripts write regenerated figures here |

## Map from paper to scripts

| paper item | script | archived output |
|---|---|---|
| Tables 1–3 (static, convergence) and 4–6 (moving target) | `code/hn_periph3.py`–`hn_periph6.py` | `results/hn_periph3.json`–`hn_periph6.json` |
| Figure 1 (sign of selection over the a×ψ plane), Table of thresholds, per-run statistics | `code/hn_periph8.py`, `hn_periph8_stats.py`, `hn_periph8_fig.py` | `results/hn_periph8*.json`, `figures/fig26_sign_boundary.*` |
| Figure 2 (seven regimes), Figure 3 (Rogers' paradox), Figure 4 (drift test) and their tables | `code/sl.py`, `code/sl_figs_en.py` | `results/sl_tables_en.json`, `figures/fig2{3,4,5}_*_en.*` |
| Population `?`-frequency for the long runs | `code/sl_pq_en.py` | printed to stdout |

Scripts are run from inside `code/` and write JSON to the repository root
and figures to `../figs/`:

```bash
cd code
python hn_periph8.py && python hn_periph8_stats.py && python hn_periph8_fig.py
python sl_figs_en.py
```

## Reproducibility note

All archived outputs were produced with **Python 3.13 / NumPy 2.4.3** under
fixed random seeds. NumPy's random streams differ between versions, so other
NumPy versions reproduce the qualitative results but not the exact digits.

Some scripts carry working notes in Vietnamese; the code and all
paper-facing outputs are in English.

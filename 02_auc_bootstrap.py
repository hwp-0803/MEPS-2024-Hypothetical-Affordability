"""Supplementary discrimination analysis (manuscript: Appendix Text A1).
Usage: python 02_auc_bootstrap.py path/to/input.dta
Outcome: family medical-bill payment problem (PROBPY42 = 1). Weighted AUCs use ESAQWT24F.
(1) AUC of the four-level payment-strategy item alone, by race/ethnicity.
(2) Pooled logistic models (covariates + race + nativity) with vs without payment-strategy indicators:
    out-of-fold AUC from 10-fold cross-validation assigning whole stratum-PSU clusters to folds, 10 repeats.
(3) Rao-Wu rescaled bootstrap (500 replicates, PSUs resampled within strata) for between-group differences
    in item-alone AUC and in the in-sample AUC gain from adding the item.
(4) For comparison with an earlier pipeline: models fitted separately within each racial/ethnic group.
"""
import sys, numpy as np, pandas as pd
SEED = 20261111
cols = ['DUPERSID','VARSTR','VARPSU','ESAQWT24F','BORNUSA','RACETHX','EQPAYB53','PROBPY42','MEDDEBT42','EQSNON53','AGE24X','SEX','POVCAT24','INSCOV24']
d = pd.read_stata(sys.argv[1], columns=cols, convert_categoricals=False)
d['psu'] = d.VARSTR.astype(int) * 100 + d.VARPSU.astype(int)
strata_all = d[d.ESAQWT24F > 0].groupby('VARSTR').psu.unique()
c = d[(d.ESAQWT24F > 0) & d.BORNUSA.isin([1, 2]) & d.RACETHX.isin([1, 2, 3, 4]) & d.EQPAYB53.between(1, 7)
      & d.PROBPY42.isin([1, 2]) & d.MEDDEBT42.between(0, 7) & d.EQSNON53.isin([1, 2])].reset_index(drop=True)
print('analytic n =', len(c))
n = len(c); w0 = c.ESAQWT24F.to_numpy(float); race = c.RACETHX.to_numpy(); fb = (c.BORNUSA == 2).to_numpy()
y = (c.PROBPY42 == 1).to_numpy(float)
pay4 = c.EQPAYB53.map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 3}).to_numpy(float)
age = (c.AGE24X.to_numpy(float) - 50) / 20
cov = [np.ones(n), age, age ** 2, c.SEX.to_numpy() == 2] + [c.POVCAT24.to_numpy() == k for k in (1, 2, 3, 4)] + \
      [c.INSCOV24.to_numpy() == 2, c.INSCOV24.to_numpy() == 3]
racecols = [race == 1, race == 3, race == 4, fb]
items = [pay4 == 1, pay4 == 2, pay4 == 3]
Xb = np.column_stack(cov + racecols).astype(float); Xf = np.column_stack(cov + racecols + items).astype(float)
Xb_g = np.column_stack(cov + [fb]).astype(float); Xf_g = np.column_stack(cov + [fb] + items).astype(float)
G = {'NHW': race == 2, 'Hisp': race == 1, 'NHB': race == 3}

def wauc(s, yy, ww):
    o = np.argsort(s, kind='mergesort'); s, yy, ww = s[o], yy[o], ww[o]
    _, inv = np.unique(s, return_inverse=True)
    gn = np.bincount(inv, ww * (1 - yy)); gp = np.bincount(inv, ww * yy); below = np.cumsum(gn) - gn
    return (gp * (below + 0.5 * gn)).sum() / ((ww * yy).sum() * (ww * (1 - yy)).sum())
def fit(X, yy, ww):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); step = np.linalg.solve((X * (ww * p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ (ww * (yy - p)))
        b += step
        if np.abs(step).max() < 1e-9: break
    return b
def stats(w):
    sb = Xb @ fit(Xb, y, w); sf = Xf @ fit(Xf, y, w); r = {}
    for g, m in G.items():
        r[g, 'item'] = wauc(pay4[m], y[m], w[m]); r[g, 'gain'] = wauc(sf[m], y[m], w[m]) - wauc(sb[m], y[m], w[m])
    for g in ('Hisp', 'NHB'):
        for k in ('item', 'gain'): r[g + '-NHW', k] = r[g, k] - r['NHW', k]
    return r

est = stats(w0)
print('\n(1) Item-alone weighted AUC:', {g: round(est[g, 'item'], 3) for g in G})
print('    In-sample AUC gain from adding item:', {g: round(est[g, 'gain'], 3) for g in G})

rng = np.random.default_rng(SEED); psus = np.unique(c.psu)
cv = {g: [] for g in G}; cvg = {g: [] for g in G}
for rep in range(10):
    fold = dict(zip(psus, rng.permutation(len(psus)) % 10)); f = np.array([fold[p] for p in c.psu])
    sb = np.zeros(n); sf = np.zeros(n); sbg = np.zeros(n); sfg = np.zeros(n)
    for k in range(10):
        tr, te = f != k, f == k
        sb[te] = Xb[te] @ fit(Xb[tr], y[tr], w0[tr]); sf[te] = Xf[te] @ fit(Xf[tr], y[tr], w0[tr])
        for g, m in G.items():
            a, t2 = tr & m, te & m
            sbg[t2] = Xb_g[t2] @ fit(Xb_g[a], y[a], w0[a]); sfg[t2] = Xf_g[t2] @ fit(Xf_g[a], y[a], w0[a])
    for g, m in G.items():
        cv[g].append((wauc(sb[m], y[m], w0[m]), wauc(sf[m], y[m], w0[m])))
        cvg[g].append((wauc(sbg[m], y[m], w0[m]), wauc(sfg[m], y[m], w0[m])))
print('\n(2) Cross-validated AUC, pooled models (mean of 10 repeats): covariates only -> + payment item')
for g in G:
    a = np.mean(cv[g], 0); print(f'    {g}: {a[0]:.3f} -> {a[1]:.3f} (gain {a[1]-a[0]:+.3f}; range of gain across repeats {min(x[1]-x[0] for x in cv[g]):+.3f} to {max(x[1]-x[0] for x in cv[g]):+.3f})')
print('(4) Cross-validated AUC, models fitted within each group (comparison pipeline)')
for g in G:
    a = np.mean(cvg[g], 0); print(f'    {g}: {a[0]:.3f} -> {a[1]:.3f} (gain {a[1]-a[0]:+.3f})')

# Rao-Wu rescaled bootstrap
upsu = np.unique(d.loc[d.ESAQWT24F > 0, 'psu']); pos = {p: i for i, p in enumerate(upsu)}; idx = np.array([pos[p] for p in c.psu])
reps = []
for b in range(500):
    mult = np.zeros(len(upsu))
    for h, ps in strata_all.items():
        nh = len(ps); cnt = np.bincount(rng.choice(nh, nh - 1, replace=True), minlength=nh)
        mult[[pos[p] for p in ps]] = cnt * nh / (nh - 1)
    reps.append(stats(w0 * mult[idx]))
R = pd.DataFrame(reps)
print('\n(3) Rao-Wu bootstrap (500 replicates), 95% percentile intervals')
for k in [('Hisp-NHW', 'item'), ('NHB-NHW', 'item'), ('Hisp-NHW', 'gain'), ('NHB-NHW', 'gain')]:
    lo, hi = np.percentile(R[k], [2.5, 97.5]); print(f'    {k[0]} {k[1]:4s}: {est[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]')

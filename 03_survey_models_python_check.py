"""Independent Python check of 01_survey_models.R (Table 2, Table 3 intervals, and Appendix Table A1 = "Table 4") without R.
Usage: python 03_survey_models_python_check.py path/to/input.dta [path/to/01_survey_models.log]
Implements the same estimators as R survey: weighted logistic regression with Taylor-linearized
(stratified, with-replacement PSU) variance; design df = #PSUs - #strata among PSUs that contain
domain observations (survey::degf for a subset design); svyciprop(method = "logit") intervals for
proportions. If the R log is supplied, every printed estimate is compared with it.
"""
import sys, re, numpy as np, pandas as pd
from scipy import stats
cols = ['DUPERSID','VARSTR','VARPSU','ESAQWT24F','BORNUSA','RACETHX','EQPAYB53','PROBPY42','MEDDEBT42','EQSNON53','AGE24X','SEX','POVCAT24','INSCOV24','RESP42','ESAQPROXY53']
d = pd.read_stata(sys.argv[1], columns=cols, convert_categoricals=False)
d = d[d.ESAQWT24F > 0].reset_index(drop=True)
d['psu'] = d.VARSTR.astype(int) * 1000 + d.VARPSU.astype(int)
common = (d.BORNUSA.isin([1, 2]) & d.RACETHX.isin([1, 2, 3, 4]) & d.EQPAYB53.between(1, 7) & d.PROBPY42.isin([1, 2])
          & d.MEDDEBT42.between(0, 7) & d.EQSNON53.isin([1, 2])).to_numpy()
selfr = ((d.RESP42 == 1) & (d.ESAQPROXY53 == 1)).to_numpy()
w = d.ESAQWT24F.to_numpy(float); race = d.RACETHX.to_numpy(); u = (d.EQPAYB53 == 7).to_numpy(float)
Y = {'probpy': (d.PROBPY42 == 1).to_numpy(float), 'debt': (d.MEDDEBT42 >= 1).to_numpy(float), 'sac': (d.EQSNON53 == 2).to_numpy(float)}
age = (d.AGE24X.to_numpy(float) - 50) / 20
H, B, A = (race == 1).astype(float), (race == 3).astype(float), (race == 4).astype(float)
X = np.column_stack([np.ones(len(d)), u, H, B, A, u * H, u * B, u * A, d.BORNUSA == 2, age, age ** 2, d.SEX == 2]
                    + [d.POVCAT24 == k for k in (1, 2, 3, 4)] + [d.INSCOV24 == 2, d.INSCOV24 == 3]).astype(float)
I_U, I_UH, I_UB = 1, 5, 6

def taylor_var(S, dom):
    df = pd.DataFrame(S); df['h'] = d.VARSTR.values; df['c'] = d.psu.values
    T = df.groupby(['h', 'c']).sum(); V = np.zeros((S.shape[1], S.shape[1]))
    for _, g in T.groupby(level=0):
        m = len(g)
        if m > 1: z = g.values - g.values.mean(0); V += m / (m - 1) * z.T @ z
    sub = d.loc[dom, ['VARSTR', 'psu']]
    return V, sub.psu.nunique() - sub.VARSTR.nunique()

def svylogit(y, dom):
    Xd, yd, wd = X[dom], y[dom], w[dom]; b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xd @ b)); st = np.linalg.solve((Xd * (wd * p * (1 - p))[:, None]).T @ Xd, Xd.T @ (wd * (yd - p))); b += st
        if np.abs(st).max() < 1e-10: break
    p = 1 / (1 + np.exp(-Xd @ b)); Ai = np.linalg.inv((Xd * (wd * p * (1 - p))[:, None]).T @ Xd)
    S = np.zeros_like(X); S[dom] = Xd * (wd * (yd - p))[:, None]
    V, df = taylor_var(S, dom); return b, Ai @ V @ Ai, df

def lc(b, C, idx, df):
    L = np.zeros(len(b)); L[idx] = 1; e = L @ b; s = np.sqrt(L @ C @ L); t = stats.t.ppf(.975, df)
    return np.exp([e, e - t * s, e + t * s])

def svyciprop_logit(y, dom):
    # intercept-only weighted logistic model on the domain, logit-scale t interval (survey::svyciprop, method = "logit")
    yd, wd = y[dom], w[dom]; p = (wd * yd).sum() / wd.sum(); b = np.log(p / (1 - p))
    S = np.zeros((len(d), 1)); S[dom, 0] = wd * (yd - p); V, df = taylor_var(S, dom)
    se = np.sqrt(V[0, 0]) / (wd.sum() * p * (1 - p)); t = stats.t.ppf(.975, df)
    f = lambda z: 1 / (1 + np.exp(-z)); return 100 * p, 100 * f(b - t * se), 100 * f(b + t * se)

out = []
print('== Table 3: weighted % [95% CI] (svyciprop logit)')
for y in ('probpy', 'debt', 'sac'):
    for uu in (1, 0):
        parts = []
        for code, g in ((2, 'NHW'), (1, 'Hisp'), (3, 'NHB')):
            dom = common & (race == code) & (u == uu); p, lo, hi = svyciprop_logit(Y[y], dom)
            parts.append(f'{g} {p:.1f} [{lo:.1f}, {hi:.1f}] n={dom.sum()}'); out += [round(p, 1), round(lo, 1), round(hi, 1)]
        print(f'{y:6s} u={uu}  ' + ' | '.join(parts))
for label, dom in (('Table 2: full analytic sample', common), ('Table 4: household respondent & ESAQ self-respondent', common & selfr)):
    print(f'\n== {label} (unweighted n = {dom.sum()})')
    for y in ('probpy', 'debt', 'sac'):
        b, C, df = svylogit(Y[y], dom)
        r = [lc(b, C, [I_U], df), lc(b, C, [I_U, I_UH], df), lc(b, C, [I_U, I_UB], df)]
        bb = b[[I_UH, I_UB]]; W = bb @ np.linalg.solve(C[np.ix_([I_UH, I_UB], [I_UH, I_UB])], bb); F = W / 2; pval = stats.f.sf(F, 2, df)
        ror = [lc(b, C, [I_UH], df), lc(b, C, [I_UB], df)]
        fmt = lambda v: f'{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]'
        print(f'{y:6s} aOR NHW {fmt(r[0])} | Hisp {fmt(r[1])} | NHB {fmt(r[2])} | Wald F(2,{df}) = {F:.2f}, p = {pval:.4f} | ROR Hisp {fmt(ror[0])} | ROR NHB {fmt(ror[1])}')
        for v in r + ror: out += [round(x, 2) for x in v]
        out += [round(F, 2)]
if len(sys.argv) > 2:
    log = open(sys.argv[2], encoding='utf-8').read()
    t3 = log[log.index('== Table 3'):log.index('== Table 2')]; t24 = log[log.index('== Table 2'):]
    rvals = [float(x) for x in re.findall(r'(\d+\.\d) \[(\d+\.\d), (\d+\.\d)\]', t3) for x in x]
    for line in t24.splitlines():
        if line.startswith(('probpy', 'debt', 'sac')):
            nums = re.findall(r'(\d+\.\d\d) \[(\d+\.\d\d), (\d+\.\d\d)\]', line); F = re.search(r'= (\d+\.\d\d),', line).group(1)
            rvals += [float(x) for tup in nums for x in tup] + [float(F)]
    diff = [(a, b) for a, b in zip(out, rvals) if abs(a - b) > 1e-9]
    print(f'\nCompared {len(rvals)} values with the R log: {len(rvals) - len(diff)} identical, {len(diff)} different')
    for a, b in diff: print(f'  Python {a} vs R {b}')

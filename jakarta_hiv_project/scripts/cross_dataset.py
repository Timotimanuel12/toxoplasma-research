"""
Do Makassar (182 pregnant women) and the Jakarta HIV study (186 people living with HIV) tell the same story?

Uses only the questions both datasets share, coded the same way:
  age, education (4 levels), raises/keeps a cat, soil contact, raw vegetables, undercooked/grilled poultry,
  un-boiled (well/river) water, refill-gallon water.
Not shared: cat contact and cleaning cat litter (Makassar's strongest questions; missing from the Jakarta export).

1. Within each dataset: 5-fold CV x 20 (logistic regression) on the shared questions only.
2. Transfer: train on all of one dataset, test on the other (ROC-AUC with a 1,000-sample bootstrap 95% CI).
3. Risk factors: odds ratio of each question in each dataset (adjusted for age), and a test of whether the two
   odds ratios differ (dataset x factor interaction, Wald test).
Variants for the two uncertain matches are run side by side.

Run from inside jakarta_hiv_project/:  python scripts/cross_dataset.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
OUT = os.path.join('outputs', 'cross_dataset.txt')
MAKASSAR = os.path.join('..', 'makassar_project', 'data', 'processed', 'makassar_clean.csv')
JAKARTA = os.path.join('data', 'processed', 'jakarta_hiv_clean.csv')

# Jakarta education 1-7 -> Makassar's 4 levels (highest level completed)
EDU_J = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 4}   # <=elementary, junior high, senior high, above senior high


def harmonise(poultry='raw_or_grilled', water=True):
    m = pd.read_csv(MAKASSAR)
    m = m[m['igg_borderline'] == 0]
    j = pd.read_csv(JAKARTA)
    j = j[j['igg_borderline'] == 0]
    raw_p = j['freq_raw_poultry_consumed_past_year'] > 0
    grill_p = j['freq_grilled_smoked_poultry_consumed_past_year'] > 0
    jp = {'raw_or_grilled': raw_p | grill_p, 'raw': raw_p, 'grilled': grill_p}[poultry]
    M = pd.DataFrame({'age': m['age'], 'education': m['education'], 'cat': m['raises_cat'],
                      'soil': m['soil_contact'], 'raw_vegetables': m['eats_raw_vegetables'],
                      'poultry': m['eats_chicken_satay'], 'unboiled_water': m['drinks_unboiled_water'],
                      'refill_gallon': m['drinks_gallon_water'], 'y': m['igg_positive']})
    J = pd.DataFrame({'age': j['age'], 'education': j['education'].map(EDU_J), 'cat': j['has_cat'],
                      'soil': (j['freq_soil_contact_activity_past_year'] > 0).astype(int),
                      'raw_vegetables': (j['freq_raw_veggies_consumed_past_year'] > 0).astype(int),
                      'poultry': jp.astype(int), 'unboiled_water': (j['drinking_water_source'] == 3).astype(int),
                      'refill_gallon': (j['drinking_water_source'] == 1).astype(int), 'y': j['igg_positive']})
    if not water:
        M, J = M.drop(columns='unboiled_water'), J.drop(columns='unboiled_water')
    assert M['education'].notna().all() and J['education'].notna().all()
    return M.reset_index(drop=True), J.reset_index(drop=True)


def model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def boot_auc(y, p, n=1000, seed=42):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def logit(X, y, ridge=1e-6, iters=100):
    """Logistic regression by Newton's method; returns coefficients and standard errors (intercept first)."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


lines = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
VARIANTS = [('main: poultry = raw or grilled; water = well/river', dict()),
            ('poultry = raw only', dict(poultry='raw')),
            ('poultry = grilled only', dict(poultry='grilled')),
            ('without the water question', dict(water=False))]
lines.append('=== 1-2. Prediction with the shared questions only (logistic regression, ROC-AUC)')
lines.append(f'{"variant":52} {"Makassar CV":>12} {"Jakarta CV":>11} {"train M -> test J":>26} {"train J -> test M":>26}')
for label, kw in VARIANTS:
    M, J = harmonise(**kw)
    f = [c for c in M.columns if c != 'y']
    cvm = cross_val_score(model(), M[f], M['y'], cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    cvj = cross_val_score(model(), J[f], J['y'], cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    pj = model().fit(M[f], M['y']).predict_proba(J[f])[:, 1]
    pm = model().fit(J[f], J['y']).predict_proba(M[f])[:, 1]
    mj, mm = roc_auc_score(J['y'], pj), roc_auc_score(M['y'], pm)
    cj, cm = boot_auc(J['y'].to_numpy(), pj), boot_auc(M['y'].to_numpy(), pm)
    lines.append(f'{label:52} {cvm:12.3f} {cvj:11.3f} {mj:8.3f} [{cj[0]:.3f}, {cj[1]:.3f}]   '
                 f'{mm:8.3f} [{cm[0]:.3f}, {cm[1]:.3f}]')
    if not kw:
        main_M, main_J, main_pj = M, J, pj
lines.append(f'Reference: Makassar with ALL its questions (incl. cat contact, litter) = 0.874; '
             f'Jakarta with all its questions = about 0.5.')
lines.append(f'Calibration of train M -> test J (main): mean predicted risk {main_pj.mean():.2f} vs '
             f'observed {main_J["y"].mean():.2f}')

lines += ['', '=== 3. Risk factors in each population (odds ratio adjusted for age; main variant)',
          f'{"question":16} {"% yes M":>8} {"% yes J":>8} {"OR Makassar [95% CI]":>24} {"OR Jakarta [95% CI]":>24}'
          f' {"differ? p":>10}']
for fac in ['cat', 'soil', 'raw_vegetables', 'poultry', 'unboiled_water', 'refill_gallon', 'education']:
    ors = []
    for D in (main_M, main_J):
        D = D.dropna(subset=['age'])
        b, se = logit(np.column_stack([D[fac], D['age'] / 10]), D['y'].to_numpy(float))
        ors.append((np.exp(b[1]), np.exp(b[1] - 1.96 * se[1]), np.exp(b[1] + 1.96 * se[1])))
    P = pd.concat([main_M.assign(ds=0), main_J.assign(ds=1)]).dropna(subset=['age'])
    Xp = np.column_stack([P['ds'], P['age'] / 10, P[fac], P[fac] * P['ds']])
    b, se = logit(Xp, P['y'].to_numpy(float))
    p_int = 2 * stats.norm.sf(abs(b[4] / se[4]))
    pct = (f'{main_M[fac].mean():8.0%} {main_J[fac].mean():8.0%}' if fac != 'education'
           else f'{"(1-4)":>8} {"(1-4)":>8}')
    lines.append(f'{fac:16} {pct} {ors[0][0]:7.2f} [{ors[0][1]:5.2f}, {ors[0][2]:6.2f}] '
                 f'{ors[1][0]:7.2f} [{ors[1][1]:5.2f}, {ors[1][2]:6.2f}] {p_int:10.4f}')
lines.append('(education: odds ratio per level; "differ? p" = test that the two odds ratios are equal)')

os.makedirs('outputs', exist_ok=True)
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nSaved {OUT}')

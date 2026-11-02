"""
Leakage check for toxo_max_score.ipynb, Part B (improvement steps 0-5).

In the notebook, step 4 picks the "biological/exposure factors independently linked to Toxoplasma"
ONCE, on all 13,647 training people, and then cross-validates steps 4-5 on those same people.
Every validation fold therefore helped choose the factors, so the CV scores of steps 4-5 (and their
p-values) are optimistic. The held-out test set is not affected.

This script copies Part B exactly (same data, same 1,002 test people, same 15 CV folds, same models,
preprocessing and scoring) and runs the factor selection in two ways:

  once      : selection on all training people, as in the notebook (should reproduce its numbers)
  per_fold  : selection repeated inside each CV fold, on that fold's training people only (the fix)

The final model for the test set is fitted on all training people in both modes, with factors
selected on all training people. That is the correct way to build the final model, so the test
ROC-AUCs are the same in both modes.

What can still differ between modes (notebook cell 14):
  - which step is best by CV (= the final model),
  - the decision threshold, which is picked on the out-of-fold predictions of the training people,
    and therefore the test sensitivity / specificity / TP-FN-FP-TN,
  - the paired bootstrap gain, if the final model changes.

Usage (from the repository root or anywhere):  python scripts/max_score_fixed.py [--mode once|per_fold|both]
"""
import argparse
import json
import os
import sys
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
sys.path.insert(0, 'scripts')
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

import build_bio_features as bbf

RANDOM_STATE = 42
N_REPEATS = 3
TARGET = 'target_toxo_pos'
OUT_JSON = os.path.join('outputs', 'max_score_fixed_results.json')

# Numbers printed by toxo_max_score.ipynb (cells 10, 11 and 14), used to check the reproduction
NOTEBOOK_SELECTED = ['hav_natural_infection', 'hav_antibody', 'hev_igg', 'hsv1', 'hsv2', 'toxocara_mfi_log', 'wbc_count',
                     'eosinophil_pct', 'lymphocyte_pct', 'blood_lead_log', 'blood_mercury_log', 'cotinine_log',
                     'well_or_spring_water', 'no_tap_water', 'household_size', 'crowding', 'home_rented',
                     'spanish_interview', 'not_us_citizen', 'work_food_service', 'work_construction_soil',
                     'food_insecurity', 'no_health_insurance', 'inject_drugs_ever', 'hav_vaccinated', 'missing_teeth']
NOTEBOOK_CV = [0.7227, 0.7267, 0.7256, 0.7303, 0.7346, 0.7398]
NOTEBOOK_TEST = [0.7126, 0.7121, 0.7132, 0.7143, 0.7333, 0.7297]
NOTEBOOK_FINAL = 'earlier: thr 0.142, TP/FN/FP/TN 98/52/288/564 | final (step 5): thr 0.136, 106/44/303/549 | ' \
                 'bootstrap gain +0.017 (+0.000 to +0.034)'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['once', 'per_fold', 'both'], default='both')
args = parser.parse_args()
MODES = ['once', 'per_fold'] if args.mode == 'both' else [args.mode]

# =====================================================================
# Data: identical to toxo_max_score.ipynb cells 1 and 10
# =====================================================================
assert os.path.exists(bbf.BASE_CSV) and os.path.exists(bbf.OUT_POOLED_CSV), 'prepared data files missing'
base = pd.read_csv(bbf.BASE_CSV)                     # the original 5,006-person 2013-2014 cohort
_, base_test = train_test_split(base, test_size=0.20, stratify=base[TARGET], random_state=RANDOM_STATE)
TEST_SEQN = set(base_test['SEQN'])

pooled = pd.read_csv(bbf.OUT_POOLED_CSV)
pooled['born_abroad'] = (pooled['DMDBORN4'] == 2).astype(float)
pooled['age_x_abroad'] = pooled['RIDAGEYR'] * pooled['born_abroad']
# (as in the notebook, the medians here are taken over the whole pooled file; a minor detail kept for exact reproduction)
pooled['age_x_edu'] = pooled['RIDAGEYR'] * pooled['DMDEDUC2'].fillna(pooled['DMDEDUC2'].median())
pooled['age_x_pir'] = pooled['RIDAGEYR'] * pooled['INDFMPIR'].fillna(pooled['INDFMPIR'].median())
train = pooled[~pooled['SEQN'].isin(TEST_SEQN)].reset_index(drop=True)
test = pooled[pooled['SEQN'].isin(TEST_SEQN)].set_index('SEQN').loc[base_test['SEQN']].reset_index()
assert len(test) == 1002 and (test[TARGET].to_numpy() == base_test[TARGET].to_numpy()).all()
is_h = (train['cycle'] == '2013-2014').to_numpy()
print(f'Training people: {len(train):,} (2013-2014: {is_h.sum():,}, 2009-2012: {(~is_h).sum():,});  test: {len(test):,}')


# =====================================================================
# Step 4 factor selection: same rule as the notebook, but for any set of training rows
# =====================================================================
def logit_fit(X, y, iters=50):
    X = np.column_stack([np.ones(len(X)), X]); beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p)); beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta[1:], np.sqrt(np.diag(np.linalg.inv(H)))[1:]


def select_factors(rows):
    """New factors still significant (p < 0.05) after age/sex/birthplace adjustment, using only `rows`."""
    d = train.iloc[rows]
    adj = np.column_stack([d['RIDAGEYR'] / 10, (d['RIAGENDR'] == 1), d['born_abroad']]).astype(float)
    y = d[TARGET].to_numpy(float)
    selected = []
    for col in bbf.NEW_FEATURES:
        x = d[col]
        xs = x if set(x.dropna().unique()) <= {0.0, 1.0} else (x - x.mean()) / x.std()
        m = x.notna().to_numpy()
        b, se = logit_fit(np.column_stack([xs[m], adj[m]]), y[m])
        if 2 * stats.norm.sf(abs(b[0] / se[0])) < 0.05:
            selected.append(col)
    return selected


# =====================================================================
# Steps and models: identical to toxo_max_score.ipynb cell 11
# =====================================================================
CAT_P = ['RIAGENDR', 'RIDRETH1', 'yrs_in_us_grp', 'SLQ050', 'smoking_status', 'toxocara_pos', 'cycle']
OLD_NUM = ['DMDEDUC2', 'INDFMPIR', 'HSD010', 'SLD010H', 'drinks_per_day', 'phq9_total_score']
INTER = ['born_abroad', 'age_x_abroad', 'age_x_edu', 'age_x_pir']


def preprocessor(cat, num, spline_age):
    steps = [('cat', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                               ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat),
             ('age', SplineTransformer(n_knots=6, degree=3) if spline_age else StandardScaler(), ['RIDAGEYR'])]
    if num:
        steps.append(('num', Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True)),
                                       ('s', StandardScaler())]), num))
    return ColumnTransformer(steps)


def xgb_model():
    return XGBClassifier(n_estimators=800, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.6,
                         min_child_weight=10, reg_lambda=5, eval_metric='logloss', n_jobs=1, random_state=RANDOM_STATE)


LEARNERS = {'lr1': lambda: LogisticRegression(C=1.0, max_iter=5000),
            'lr03': lambda: LogisticRegression(C=0.3, max_iter=5000), 'xgb': xgb_model}
STEP_NAMES = ['0 earlier best (2013-14 only)', '1 + 2009-2014 training rows', '2 + all earlier features, age spline',
              '3 + age interactions', '4 + selected bio/exposure factors', '5 + XGBoost ensemble']
USES_SELECTION = {'4 + selected bio/exposure factors', '5 + XGBoost ensemble'}


def step_spec(name, selected):
    """(training scope, categorical, numeric, spline age, learners) for a step, given the selected factors."""
    return {
        STEP_NAMES[0]: ('H', ['RIAGENDR', 'RIDRETH3', 'yrs_in_us_grp', 'toxocara_pos'], ['DMDEDUC2', 'INDFMPIR'], False, ['lr1']),
        STEP_NAMES[1]: ('all', ['RIAGENDR', 'RIDRETH1', 'yrs_in_us_grp', 'toxocara_pos', 'cycle'], ['DMDEDUC2', 'INDFMPIR'], False, ['lr1']),
        STEP_NAMES[2]: ('all', CAT_P, OLD_NUM, True, ['lr1']),
        STEP_NAMES[3]: ('all', CAT_P, OLD_NUM + INTER, True, ['lr1']),
        STEP_NAMES[4]: ('all', CAT_P, OLD_NUM + INTER + selected, True, ['lr03']),
        STEP_NAMES[5]: ('all', CAT_P, OLD_NUM + INTER + selected, True, ['lr03', 'xgb']),
    }[name]


def fit_step(name, fit_rows, selected):
    scope, cat, num, spline, learners = step_spec(name, selected)
    rows = fit_rows if scope == 'all' else fit_rows[train.loc[fit_rows, 'cycle'].to_numpy() == '2013-2014']
    cols = cat + num + ['RIDAGEYR']
    pipes = []
    for l in learners:
        if l == 'xgb':
            pre = ColumnTransformer([('cat', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                                                       ('o', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat),
                                     ('num', 'passthrough', num + ['RIDAGEYR'])])
        else:
            pre = preprocessor(cat, num, spline)
        pipes.append(make_pipeline(pre, LEARNERS[l]()).fit(train.loc[rows, cols], train.loc[rows, TARGET]))
    return lambda df: np.mean([p.predict_proba(df[cols])[:, 1] for p in pipes], axis=0)


def corrected_p(d, ratio=0.25):
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
    return 2 * stats.t.sf(abs(t), df=len(d) - 1)


# =====================================================================
# Cross-validation: same 15 folds as the notebook
# =====================================================================
CV = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(train, train[TARGET]))
ALL_ROWS = np.arange(len(train))
SELECTED_ALL = select_factors(ALL_ROWS)                  # notebook's selection (all training people)
print(f'Factors selected on all training people ({len(SELECTED_ALL)}):', ', '.join(SELECTED_ALL))
if SELECTED_ALL != NOTEBOOK_SELECTED:
    sys.exit('STOP: selection differs from the notebook.\n'
             f'  missing: {sorted(set(NOTEBOOK_SELECTED) - set(SELECTED_ALL))}\n'
             f'  extra:   {sorted(set(SELECTED_ALL) - set(NOTEBOOK_SELECTED))}')
print('  -> identical to the notebook (same 26 factors, same order)')

# Selection inside each fold (only needed for the per_fold mode)
fold_selected = None
if 'per_fold' in MODES:
    fold_selected = Parallel(n_jobs=-1)(delayed(select_factors)(tr) for tr, _ in CV)
    counts = [len(s) for s in fold_selected]
    print(f'Factors selected inside each fold: min {min(counts)}, max {max(counts)}, mean {np.mean(counts):.1f}')


def cv_job(mode, name, k, tr, va, selected):
    predict = fit_step(name, tr, selected)
    va_h = va[is_h[va]]
    return mode, name, k, va_h, predict(train.loc[va_h])


jobs = []
for name in STEP_NAMES:
    if name not in USES_SELECTION:                       # steps 0-3 do not depend on the selection
        jobs += [('shared', name, k, tr, va, []) for k, (tr, va) in enumerate(CV)]
    else:
        for mode in MODES:
            jobs += [(mode, name, k, tr, va, SELECTED_ALL if mode == 'once' else fold_selected[k])
                     for k, (tr, va) in enumerate(CV)]
print(f'Fitting {len(jobs)} fold models ...', flush=True)
out = Parallel(n_jobs=-1)(delayed(cv_job)(*j) for j in jobs)

# CV ROC-AUC per fold, and out-of-fold predictions averaged over the repeats (as in the notebook)
cv_auc = {mode: {n: np.zeros(len(CV)) for n in STEP_NAMES} for mode in MODES}
oof = {mode: {n: np.zeros(len(train)) for n in STEP_NAMES} for mode in MODES}
for mode, name, k, va_h, p in out:
    auc = roc_auc_score(train.loc[va_h, TARGET], p)
    for m in (MODES if mode == 'shared' else [mode]):
        cv_auc[m][name][k] = auc
        oof[m][name][va_h] += p / N_REPEATS

# =====================================================================
# Held-out test set: final models fitted on all training people (same in both modes)
# =====================================================================
y_test = test[TARGET].to_numpy()
test_p = {n: fit_step(n, ALL_ROWS, SELECTED_ALL)(test) for n in STEP_NAMES}
test_auc = {n: roc_auc_score(y_test, test_p[n]) for n in STEP_NAMES}


def thr_from_oof(o):
    """Threshold maximising balanced accuracy on the 2013-2014 training people's out-of-fold predictions."""
    m = is_h & (o > 0)
    fpr, tpr, thr = roc_curve(train.loc[m, TARGET], o[m])
    return thr[np.argmax(tpr - fpr)]


def test_metrics(p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {'threshold': float(thr), 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'balanced accuracy': balanced_accuracy_score(y_test, pred),
            'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)}


def paired_bootstrap(p_new, p_old):
    """Test-set ROC-AUC gain of p_new over p_old, 1,000 paired resamples (same seed as the notebook)."""
    rng = np.random.default_rng(RANDOM_STATE)
    d = []
    while len(d) < 1000:
        i = rng.integers(0, len(y_test), len(y_test))
        if 0 < y_test[i].sum() < len(i):
            d.append(roc_auc_score(y_test[i], p_new[i]) - roc_auc_score(y_test[i], p_old[i]))
    d = np.array(d)
    return {'gain': d.mean(), 'ci_low': np.percentile(d, 2.5), 'ci_high': np.percentile(d, 97.5),
            'share_better': np.mean(d > 0)}

# =====================================================================
# Report
# =====================================================================
results = {'selected_all_training': SELECTED_ALL}
for mode in MODES:
    base_scores = cv_auc[mode][STEP_NAMES[0]]
    rows = []
    for n in STEP_NAMES:
        d = cv_auc[mode][n] - base_scores
        rows.append({'step': n, 'CV ROC-AUC': cv_auc[mode][n].mean(), 'CV sd': cv_auc[mode][n].std(),
                     'gain vs step 0': d.mean(), 'p vs step 0': corrected_p(d) if n != STEP_NAMES[0] else np.nan,
                     'TEST ROC-AUC': test_auc[n]})
    table = pd.DataFrame(rows)
    if mode == 'once':
        table['notebook CV'], table['notebook TEST'] = NOTEBOOK_CV, NOTEBOOK_TEST
    final = max(STEP_NAMES[1:], key=lambda n: cv_auc[mode][n].mean())
    print(f'\n=== Mode: {mode} ({"selection on all training people, as in the notebook" if mode == "once" else "selection inside each fold"})')
    print(table.round(4).to_string(index=False))
    print(f'Best step by CV (= final model): {final}')

    # Notebook cell 14: threshold from out-of-fold predictions, applied unchanged to the test set
    old = STEP_NAMES[0]
    final_table = pd.DataFrame({'earlier model': test_metrics(test_p[old], thr_from_oof(oof[mode][old])),
                                'final model': test_metrics(test_p[final], thr_from_oof(oof[mode][final]))}).T
    boot = paired_bootstrap(test_p[final], test_p[old])
    print(final_table.round(3).to_string())
    print(f'Paired bootstrap test gain (final - earlier): {boot["gain"]:+.3f} '
          f'(95% CI {boot["ci_low"]:+.3f} to {boot["ci_high"]:+.3f}; final better in {boot["share_better"]:.0%})')
    if mode == 'once':
        print(f'Notebook: {NOTEBOOK_FINAL}')
    results[mode] = {'table': table.round(6).to_dict('records'), 'best_step_by_cv': final,
                     'test_at_threshold': final_table.to_dict('index'), 'bootstrap': boot,
                     'cv_auc_per_fold': {n: cv_auc[mode][n].round(6).tolist() for n in STEP_NAMES}}

if fold_selected is not None:
    freq = pd.Series([c for s in fold_selected for c in s]).value_counts().reindex(bbf.NEW_FEATURES, fill_value=0)
    freq = freq[freq > 0].sort_values(ascending=False)
    print('\nHow often each factor was selected across the 15 folds (selected on all training people = *):')
    for c, v in freq.items():
        print(f'  {c:32} {v:2d}/15 {"*" if c in SELECTED_ALL else ""}')
    results['per_fold_selection'] = {'counts_per_fold': [len(s) for s in fold_selected],
                                     'frequency': freq.to_dict(), 'folds': fold_selected}

os.makedirs('outputs', exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=1)
print(f'\nSaved {OUT_JSON}')

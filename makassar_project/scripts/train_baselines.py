"""
Baseline models for the Makassar data (182 pregnant women, 60 IgG positive; borderline excluded).

Steps:
  1. Load the clean data
  2. Cross-validate 4 models x 3 feature sets (5-fold CV repeated 20 times = 100 folds, same folds for all)
  3. Compare: models vs dummy, exposures vs background, adding background, complex models vs LR
  4. Site check: leave-one-site-out (train on 8 health centres, test on the 9th)

Run from inside makassar_project/. Per-fold scores are saved so later experiments
(self-distillation) can be compared on exactly the same folds.
"""
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import FEATURE_SETS, OUT_DIR, TorchMLP, corrected_ttest, load_data, make_cv_splits, make_preprocessor

# =====================================================================
# STEP 1: Load the data
# =====================================================================
df = load_data()
y = df['igg_positive']
print(f'Data: {len(df)} women, {y.sum()} positive ({y.mean():.1%}), {df["site"].nunique()} health centres')

# Fixed settings (no tuning), as in the NHANES project
MODELS = {
    'Dummy': DummyClassifier(strategy='prior'),
    'LogReg': LogisticRegression(C=1.0, max_iter=5000),
    'RandomForest': RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1, random_state=42),
    'MLP': TorchMLP(hidden=16, weight_decay=1e-3, lr=0.01, epochs=300, random_state=0),
}
REAL_MODELS = ['LogReg', 'RandomForest', 'XGBoost', 'MLP']


def pipeline(model_name, set_name):
    return Pipeline([('prepare', make_preprocessor(FEATURE_SETS[set_name])), ('model', MODELS[model_name])])


# =====================================================================
# STEP 2: Cross-validation
# =====================================================================
splits = make_cv_splits(df, y)
print(f'Cross-validation: {len(splits)} folds (5-fold x 20 repeats)\n')
print(f'{"Model":13} {"Set":13} {"ROC-AUC":>15} {"PR-AUC":>15} {"Brier":>8}')

scores = {}            # (model, set) -> per-fold ROC-AUC
fold_rows = []
for set_name in FEATURE_SETS:
    for model_name in MODELS:
        if model_name == 'Dummy' and set_name != 'A_background':
            continue   # the dummy ignores the features, so one run is enough
        cols = FEATURE_SETS[set_name]
        res = cross_validate(pipeline(model_name, set_name), df[cols], y, cv=splits,
                             scoring=['roc_auc', 'average_precision', 'neg_brier_score'])
        roc, pr, brier = res['test_roc_auc'], res['test_average_precision'], -res['test_neg_brier_score']
        scores[(model_name, set_name)] = roc
        print(f'{model_name:13} {set_name:13} {roc.mean():.3f} +/- {roc.std():.3f} '
              f'{pr.mean():.3f} +/- {pr.std():.3f} {brier.mean():8.4f}')
        for i in range(len(roc)):
            fold_rows.append({'model': model_name, 'set': set_name, 'fold': i,
                              'roc_auc': roc[i], 'pr_auc': pr[i], 'brier': brier[i]})

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, 'baseline_fold_scores.csv'), index=False)


# =====================================================================
# STEP 3: Comparisons (meaningful = ROC-AUC gain >= 0.01 AND corrected p < 0.05)
# =====================================================================
def compare(base, other, label):
    diff, p = corrected_ttest(scores[base], scores[other])
    verdict = 'MEANINGFUL' if diff >= 0.01 and p < 0.05 else 'not meaningful'
    print(f'  {label:42} diff {diff:+.4f}  p={p:.3f}  -> {verdict}')


print('\nComparisons (meaningful = ROC-AUC gain >= 0.01 AND corrected p < 0.05)')
print('Q1: models beat the dummy (all features)')
for m in REAL_MODELS:
    compare(('Dummy', 'A_background'), (m, 'C_all'), m + ' vs Dummy')
print('Q2: exposures beat background (B vs A)')
for m in REAL_MODELS:
    compare((m, 'A_background'), (m, 'B_exposures'), m + ': B vs A')
print('Q3: adding background to exposures helps (C vs B)')
for m in REAL_MODELS:
    compare((m, 'B_exposures'), (m, 'C_all'), m + ': C vs B')
print('Q4: complex models beat logistic regression (all features)')
for m in ['RandomForest', 'XGBoost', 'MLP']:
    compare(('LogReg', 'C_all'), (m, 'C_all'), m + ' vs LogReg')


# =====================================================================
# STEP 4: Site check, leave-one-site-out (train on 8 health centres, test on the 9th)
# =====================================================================
# Small sites give unstable per-site scores, so the main number is the pooled ROC-AUC:
# every woman gets a prediction from a model that never saw her health centre.
print('\nLeave-one-site-out (9 health centres)')
logo = LeaveOneGroupOut()
for set_name in ['B_exposures', 'C_all']:
    cols = FEATURE_SETS[set_name]
    for m in REAL_MODELS:
        p = cross_val_predict(pipeline(m, set_name), df[cols], y, groups=df['site'], cv=logo,
                              method='predict_proba')[:, 1]
        per_site = [roc_auc_score(y[df['site'] == s], p[df['site'] == s]) for s in sorted(df['site'].unique())]
        print(f'  {m:13} {set_name:13} pooled ROC-AUC {roc_auc_score(y, p):.3f}   '
              f'per site {min(per_site):.2f}-{max(per_site):.2f} (median {np.median(per_site):.2f})')

print(f'\nPer-fold scores saved to {os.path.join(OUT_DIR, "baseline_fold_scores.csv")}')

"""
Step 4: self-distillation (SD), feature set C, same 100 folds. The teacher is always the same
kind of model as the student (MLP teaches MLP, LR teaches LR).

Inside each training fold:
  1. Teacher soft labels: the training fold is split into 5 inner parts; each part gets its
     probabilities from a teacher trained on the other 4 parts, so no woman is labelled by a
     teacher that saw her (out-of-fold). The test fold is never used.
  2. Temperature T softens the teacher: p_T = sigmoid(logit(p) / T)  (T = 1: unchanged).
  3. Student target = alpha * true label + (1 - alpha) * p_T; a fresh student trains on it.
  4. Only the student is scored, on the test fold, against the true labels.

Conditions (fixed before running; every result is reported):
  - SD, MLP teaches MLP:  alpha in {0.3, 0.5, 0.7} x T in {1, 2, 4}, 1 generation
  - SD, LR teaches LR:    same grid
  - Generations 2 and 3:  only alpha = 0.5, T = 2 (the student of generation g is the teacher of g+1)

Comparisons use the per-fold scores of step 2/3 (same folds):
  - each condition vs the same student with hard labels (baseline)
  - MLP conditions also vs the strongest control from step 3 (MLP with weight decay 0.1)
Rules: ROC-AUC meaningful = gain >= 0.01 AND corrected p < 0.05; WORSE = lower AND p < 0.05.
       Brier better = lower AND p < 0.05; WORSE = higher AND p < 0.05.

Note: for generations 2-3, the earlier generation's targets come from the whole training fold,
so a woman's own label can reach her later soft label indirectly. The test fold stays untouched.

Run from inside makassar_project/.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from common import (FEATURE_SETS, OUT_DIR, SoftLabelLogReg, TorchMLP, corrected_ttest, evaluate,
                    load_data, make_cv_splits, make_preprocessor)

# MLP training settings (same as steps 2-3 unless step 4a showed they must change)
MLP_SETTINGS = dict(hidden=16, weight_decay=1e-3, lr=0.01, epochs=300)

df = load_data()
y = df['igg_positive'].to_numpy()
cols = FEATURE_SETS['C_all']
X = df[cols]
splits = make_cv_splits(df, y)


def mlp():
    return Pipeline([('prepare', make_preprocessor(cols)), ('model', TorchMLP(**MLP_SETTINGS))])


def logreg():
    return Pipeline([('prepare', make_preprocessor(cols)), ('model', SoftLabelLogReg(C=1.0))])


def temper(p, T):
    """Soften probabilities with temperature T (T = 1 leaves them unchanged)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))


def out_of_fold_probs(make_teacher, X_train, targets, y_train, seed=0):
    """Teacher probabilities for every training woman, each from a teacher that did not see her."""
    p = np.zeros(len(targets))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in inner.split(X_train, y_train):
        p[te] = make_teacher().fit(X_train.iloc[tr], targets[tr]).predict_proba(X_train.iloc[te])[:, 1]
    return p


def distil(make_teacher, alpha, T, generations=1):
    """Target function for evaluate(): soft labels built from the training fold only."""
    def make_targets(X_train, y_train):
        targets = y_train.astype(float)
        for _ in range(generations):
            p = out_of_fold_probs(make_teacher, X_train, targets, y_train)
            targets = alpha * y_train + (1 - alpha) * temper(p, T)
        return targets
    return make_targets


GRID = list(itertools.product([0.3, 0.5, 0.7], [1, 2, 4]))
EXPERIMENTS = []          # (student, condition label, model factory, target function)
for a, T in GRID:
    EXPERIMENTS.append(('MLP', f'SD  alpha={a} T={T}', mlp, distil(mlp, a, T)))
for g in [2, 3]:
    EXPERIMENTS.append(('MLP', f'SD  alpha=0.5 T=2 gen={g}', mlp, distil(mlp, 0.5, 2, g)))
for a, T in GRID:
    EXPERIMENTS.append(('LogReg', f'SD  alpha={a} T={T}', logreg, distil(logreg, a, T)))
for g in [2, 3]:
    EXPERIMENTS.append(('LogReg', f'SD  alpha=0.5 T=2 gen={g}', logreg, distil(logreg, 0.5, 2, g)))

# Per-fold scores of the comparison conditions from step 3 (identical folds)
ctrl = pd.read_csv(os.path.join(OUT_DIR, 'controls_fold_scores.csv'))


def fold_scores(student, condition, metric):
    s = ctrl[(ctrl.student == student) & (ctrl.condition == condition)].sort_values('fold')
    return s[metric].to_numpy()


print(f'Data: {len(df)} women, {y.sum()} positive; feature set C; {len(splits)} folds; MLP settings {MLP_SETTINGS}\n')
print(f'{"Student":7} {"Condition":30} {"ROC-AUC":>7} {"PR-AUC":>7} {"Brier":>7} {"Slope":>6} | '
      f'{"vs hard labels: ROC, Brier":34} | {"vs MLP weight decay 0.1: ROC, Brier"}')


def verdicts(base_roc, base_br, r):
    d_roc, p_roc = corrected_ttest(base_roc, r['roc_auc'])
    d_br, p_br = corrected_ttest(base_br, r['brier'])
    v_roc = 'MEANINGFUL' if d_roc >= 0.01 and p_roc < 0.05 else ('WORSE' if d_roc < 0 and p_roc < 0.05 else 'ns')
    v_br = 'BETTER' if d_br < 0 and p_br < 0.05 else ('WORSE' if d_br > 0 and p_br < 0.05 else 'ns')
    return f'{d_roc:+.3f} {v_roc:10} {d_br:+.4f} {v_br:6}'


rows = []
for student, cond, make_model, make_targets in EXPERIMENTS:
    t0 = time.time()
    r = evaluate(make_model, X, y, splits, make_targets)
    base = 'baseline (hard labels)'
    line = (f'{student:7} {cond:30} {r["roc_auc"].mean():7.3f} {r["pr_auc"].mean():7.3f} '
            f'{r["brier"].mean():7.4f} {r["cal_slope"].mean():6.2f} | '
            f'{verdicts(fold_scores(student, base, "roc_auc"), fold_scores(student, base, "brier"), r)} | ')
    if student == 'MLP':
        line += verdicts(fold_scores('MLP', 'L2 weight decay=0.1', 'roc_auc'),
                         fold_scores('MLP', 'L2 weight decay=0.1', 'brier'), r)
    print(line + f'  ({time.time() - t0:.0f}s)', flush=True)
    rows += [{'student': student, 'condition': cond, 'fold': i, 'roc_auc': r['roc_auc'][i],
              'pr_auc': r['pr_auc'][i], 'brier': r['brier'][i]} for i in range(len(r['roc_auc']))]

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'self_distillation_fold_scores.csv'), index=False)
print(f'\nPer-fold scores saved to {os.path.join(OUT_DIR, "self_distillation_fold_scores.csv")}')

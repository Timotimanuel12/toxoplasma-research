"""
Step 3: controls for self-distillation (feature set C, same 100 folds as the baselines).

Self-distillation softens the training targets, which on its own acts like extra regularisation.
These controls measure what plain regularisation achieves, so step 4 can show whether the
teacher's per-person soft labels add anything beyond it.

  - Label smoothing: positives -> 1 - eps, negatives -> eps (same for everyone, no teacher)
  - Stronger L2:     MLP weight decay 0.01 / 0.1 (baseline 0.001); LR C = 0.1 / 0.01 (baseline 1)

Students: MLP (main) and logistic regression. Every setting is reported (no picking the best).

Rules: ROC-AUC meaningful = gain >= 0.01 AND corrected p < 0.05; WORSE = lower AND p < 0.05.
       Brier better = lower AND corrected p < 0.05; WORSE = higher AND p < 0.05.

Run from inside makassar_project/.
"""
import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from common import (FEATURE_SETS, OUT_DIR, SoftLabelLogReg, TorchMLP, corrected_ttest, evaluate,
                    load_data, make_cv_splits, make_preprocessor)

df = load_data()
y = df['igg_positive'].to_numpy()
cols = FEATURE_SETS['C_all']
X = df[cols]
splits = make_cv_splits(df, y)          # identical to the baseline folds (same seed)
print(f'Data: {len(df)} women, {y.sum()} positive; feature set C; {len(splits)} folds\n')


def mlp(weight_decay=1e-3):
    return lambda: Pipeline([('prepare', make_preprocessor(cols)),
                             ('model', TorchMLP(hidden=16, weight_decay=weight_decay, lr=0.01, epochs=300))])


def logreg(C=1.0):
    return lambda: Pipeline([('prepare', make_preprocessor(cols)), ('model', SoftLabelLogReg(C=C))])


def smooth(eps):
    return lambda X_train, y_train: y_train * (1 - eps) + (1 - y_train) * eps


# (student, condition) -> (model factory, target function)
EXPERIMENTS = {
    ('MLP', 'baseline (hard labels)'): (mlp(), None),
    ('MLP', 'label smoothing eps=0.05'): (mlp(), smooth(0.05)),
    ('MLP', 'label smoothing eps=0.10'): (mlp(), smooth(0.10)),
    ('MLP', 'label smoothing eps=0.20'): (mlp(), smooth(0.20)),
    ('MLP', 'L2 weight decay=0.01'): (mlp(0.01), None),
    ('MLP', 'L2 weight decay=0.1'): (mlp(0.1), None),
    ('LogReg', 'baseline (hard labels)'): (logreg(), None),
    ('LogReg', 'label smoothing eps=0.05'): (logreg(), smooth(0.05)),
    ('LogReg', 'label smoothing eps=0.10'): (logreg(), smooth(0.10)),
    ('LogReg', 'label smoothing eps=0.20'): (logreg(), smooth(0.20)),
    ('LogReg', 'L2 C=0.1'): (logreg(0.1), None),
    ('LogReg', 'L2 C=0.01'): (logreg(0.01), None),
}

results = {}
print(f'{"Student":8} {"Condition":28} {"ROC-AUC":>15} {"PR-AUC":>8} {"Brier":>8} {"Cal. slope":>11}')
for (student, cond), (make_model, make_targets) in EXPERIMENTS.items():
    r = evaluate(make_model, X, y, splits, make_targets)
    results[(student, cond)] = r
    print(f'{student:8} {cond:28} {r["roc_auc"].mean():.3f} +/- {r["roc_auc"].std():.3f} '
          f'{r["pr_auc"].mean():8.3f} {r["brier"].mean():8.4f} {r["cal_slope"].mean():11.2f}')

print('\nEach control vs the same student with hard labels')
for (student, cond), r in results.items():
    if cond.startswith('baseline'):
        continue
    base = results[(student, 'baseline (hard labels)')]
    d_roc, p_roc = corrected_ttest(base['roc_auc'], r['roc_auc'])
    d_br, p_br = corrected_ttest(base['brier'], r['brier'])
    roc_v = ('MEANINGFUL' if d_roc >= 0.01 and p_roc < 0.05 else
             'WORSE' if d_roc < 0 and p_roc < 0.05 else 'not meaningful')
    br_v = 'BETTER' if d_br < 0 and p_br < 0.05 else ('WORSE' if d_br > 0 and p_br < 0.05 else 'no difference')
    print(f'  {student:7} {cond:28} ROC {d_roc:+.4f} p={p_roc:.3f} {roc_v:15} Brier {d_br:+.4f} p={p_br:.3f} {br_v}')

rows = [{'student': s, 'condition': c, 'fold': i, 'roc_auc': r['roc_auc'][i], 'pr_auc': r['pr_auc'][i],
         'brier': r['brier'][i]} for (s, c), r in results.items() for i in range(len(r['roc_auc']))]
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'controls_fold_scores.csv'), index=False)
print(f'\nPer-fold scores saved to {os.path.join(OUT_DIR, "controls_fold_scores.csv")}')

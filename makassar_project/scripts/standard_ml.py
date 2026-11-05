"""
Standard machine-learning prediction of Toxoplasma IgG status in the Makassar data, reported the same way as
jakarta_hiv_project/scripts/standard_ml.py so the two datasets can be compared side by side.

Six simple models + two dummy baselines, all questions (feature set C_all), the 2 borderline women excluded.
  A. 5-fold cross-validation repeated 20 times (seed 42): ROC-AUC, PR-AUC, accuracy, balanced accuracy,
     F1 (positive class), weighted F1 (threshold 0.5).
  B. Confusion matrices from one 5-fold cross-validation (every woman predicted once, by a model that never saw her).
  C. A single 70/30 split repeated with 100 different seeds: how much one split can move the result.

Run from inside makassar_project/:  python scripts/standard_ml.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURE_SETS, OUT_DIR, load_data, make_preprocessor

warnings.filterwarnings('ignore')
OUT = os.path.join(OUT_DIR, 'standard_ml_results.txt')

df = load_data()
FEATURES = FEATURE_SETS['C_all']
X, y = df[FEATURES], df['igg_positive'].to_numpy()
assert len(df) == 182 and y.sum() == 60


def pipe(model):
    return Pipeline([('prepare', make_preprocessor(FEATURES)), ('model', model)])


MODELS = {
    'Dummy: always negative': pipe(DummyClassifier(strategy='most_frequent')),
    'Dummy: random guess': pipe(DummyClassifier(strategy='stratified', random_state=0)),
    'Logistic regression': pipe(LogisticRegression(C=1.0, max_iter=5000)),
    'Decision tree': pipe(DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)),
    'Random forest': pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0)),
    'k-nearest neighbours': pipe(KNeighborsClassifier(n_neighbors=15)),
    'Naive Bayes': pipe(GaussianNB()),
    'SVM (RBF)': pipe(SVC(C=1.0, probability=True, random_state=0)),
}
SCORING = {'roc': 'roc_auc', 'pr': 'average_precision', 'acc': 'accuracy', 'bal': 'balanced_accuracy',
           'f1': make_scorer(f1_score, zero_division=0), 'f1w': make_scorer(f1_score, average='weighted')}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

lines = [f'Makassar data: {len(df)} pregnant women, {y.sum()} positive ({y.mean():.0%}), {len(FEATURES)} questions', '',
         'A. 5-fold CV x 20 repeats (mean +/- SD over 100 folds; threshold 0.5)',
         f'{"model":24} {"ROC-AUC":>14} {"PR-AUC":>7} {"accuracy":>9} {"bal.acc":>8} {"F1 pos":>7} {"F1 wtd":>7}']
for name, m in MODELS.items():
    s = cross_validate(m, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    lines.append(f'{name:24} {s["test_roc"].mean():.3f} +/- {s["test_roc"].std():.3f} {s["test_pr"].mean():7.3f} '
                 f'{s["test_acc"].mean():9.3f} {s["test_bal"].mean():8.3f} {s["test_f1"].mean():7.3f} '
                 f'{s["test_f1w"].mean():7.3f}')
lines.append('Here most women are NEGATIVE (67%), so the dummy that calls everyone negative gets 67% accuracy.')

lines += ['', 'B. Confusion matrices (one 5-fold CV, every woman predicted once, threshold 0.5)']
cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
for name in ['Logistic regression', 'Random forest', 'SVM (RBF)']:
    pred = cross_val_predict(MODELS[name], X, y, cv=cv5)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    lines.append(f'  {name:20} accuracy {(tp + tn) / len(y):.0%} | infected (60): found {tp}, missed {fn} | '
                 f'not infected (122): correct {tn}, wrongly positive {fp}')

lines += ['', 'C. One 70/30 split, repeated with 100 different seeds',
          f'{"model":24} {"ROC-AUC mean (5th-95th pct)":>30} {"accuracy mean (5th-95th pct)":>30}']
for name in ['Logistic regression', 'Random forest', 'SVM (RBF)']:
    roc, acc = [], []
    for seed in range(100):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
        m = MODELS[name].fit(Xtr, ytr)
        roc.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        acc.append((m.predict(Xte) == yte).mean())
    lines.append(f'{name:24} {np.mean(roc):.3f} ({np.percentile(roc, 5):.3f}-{np.percentile(roc, 95):.3f})'
                 f'{"":9} {np.mean(acc):.3f} ({np.percentile(acc, 5):.3f}-{np.percentile(acc, 95):.3f})')

os.makedirs(OUT_DIR, exist_ok=True)
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nSaved {OUT}')

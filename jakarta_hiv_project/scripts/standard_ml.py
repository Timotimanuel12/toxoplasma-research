"""
Standard machine-learning prediction of Toxoplasma IgG status in the Jakarta HIV data.

Six simple models + two dummy baselines, all questions from the cleaned file (mixed scales through _applies/_risk,
igg_titer never used, borderline person excluded).
  A. 5-fold cross-validation repeated 20 times (seed 42), as in the Makassar analysis.
     ROC-AUC, PR-AUC, accuracy, balanced accuracy, F1 (positive class), weighted F1 (threshold 0.5).
  B. A single 70/30 split (the design of the earlier paper), repeated with 100 different seeds,
     to show how much one split can move the result.

Run from inside jakarta_hiv_project/:  python scripts/standard_ml.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
OUT = os.path.join('outputs', 'standard_ml_results.txt')

d = pd.read_csv(os.path.join('data', 'processed', 'jakarta_hiv_clean.csv'))
dic = pd.read_csv(os.path.join('data', 'processed', 'jakarta_hiv_data_dictionary.csv')).set_index('column')
d = d[d['igg_borderline'] == 0].reset_index(drop=True)
y = d['igg_positive'].to_numpy()
FEATURES = [c for c in d.columns if dic.loc[c, 'role'] == 'feature']
assert 'igg_titer' not in FEATURES and 'igg_positive' not in FEATURES
X = d[FEATURES].copy()
for c in FEATURES:
    if c != 'age':
        X[c] = X[c].fillna(0)          # blanks mean "does not apply" (carried by the _applies flags)
for c in ['num_cats', 'freq_tooth_decay', 'freq_tooth_loss_extraction', 'freq_tooth_loss_natural_causes',
          'freq_loose_teeth', 'freq_swollen_gums_or_abscess', 'freq_mouth_ulcer_3_months']:
    X[c] = X[c].clip(upper=10)


def pipe(model, scale=True):
    steps = [SimpleImputer(strategy='median')] + ([StandardScaler()] if scale else []) + [model]
    return make_pipeline(*steps)


MODELS = {
    'Dummy: always positive': pipe(DummyClassifier(strategy='most_frequent'), scale=False),
    'Dummy: random guess': pipe(DummyClassifier(strategy='stratified', random_state=0), scale=False),
    'Logistic regression': pipe(LogisticRegression(C=0.1, max_iter=5000)),
    'Decision tree': pipe(DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0), scale=False),
    'Random forest': pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0), scale=False),
    'k-nearest neighbours': pipe(KNeighborsClassifier(n_neighbors=15)),
    'Naive Bayes': pipe(GaussianNB()),
    'SVM (RBF)': pipe(SVC(C=1.0, probability=True, random_state=0)),
}
SCORING = {'roc': 'roc_auc', 'pr': 'average_precision', 'acc': 'accuracy', 'bal': 'balanced_accuracy',
           'f1': 'f1', 'f1w': make_scorer(f1_score, average='weighted')}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

lines = [f'Jakarta HIV data: {len(d)} people, {y.sum()} positive ({y.mean():.0%}), {len(FEATURES)} features', '',
         'A. 5-fold CV x 20 repeats (mean +/- SD over 100 folds; threshold 0.5)',
         f'{"model":24} {"ROC-AUC":>14} {"PR-AUC":>7} {"accuracy":>9} {"bal.acc":>8} {"F1 pos":>7} {"F1 wtd":>7}']
for name, m in MODELS.items():
    s = cross_validate(m, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    lines.append(f'{name:24} {s["test_roc"].mean():.3f} +/- {s["test_roc"].std():.3f} {s["test_pr"].mean():7.3f} '
                 f'{s["test_acc"].mean():9.3f} {s["test_bal"].mean():8.3f} {s["test_f1"].mean():7.3f} '
                 f'{s["test_f1w"].mean():7.3f}')
lines.append('Balanced accuracy 0.5 and ROC-AUC 0.5 = no better than guessing.')

lines += ['', 'B. One 70/30 split (the earlier paper\'s design), repeated with 100 different seeds',
          f'{"model":24} {"ROC-AUC range (5th-95th pct)":>30} {"weighted F1 range":>20}']
for name in ['Dummy: always positive', 'Logistic regression', 'Random forest', 'SVM (RBF)']:
    roc, f1w = [], []
    for seed in range(100):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
        m = MODELS[name].fit(Xtr, ytr)
        roc.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        f1w.append(f1_score(yte, m.predict(Xte), average='weighted'))
    lines.append(f'{name:24} {np.mean(roc):.3f} ({np.percentile(roc, 5):.3f}-{np.percentile(roc, 95):.3f})'
                 f'{"":7} {np.mean(f1w):.3f} ({np.percentile(f1w, 5):.3f}-{np.percentile(f1w, 95):.3f})')
lines.append('A single split can land anywhere in these ranges, so one split alone cannot show a model works.')

os.makedirs('outputs', exist_ok=True)
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nSaved {OUT}')

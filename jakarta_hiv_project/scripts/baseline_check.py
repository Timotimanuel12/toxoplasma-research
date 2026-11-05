"""
How much can normal machine learning predict IgG status in the Jakarta HIV data? (cleaned file, corrected ages)

Same set-up as the Makassar baselines: 5-fold CV repeated 20 times (seed 42), ROC-AUC, PR-AUC, Brier; the borderline
person is excluded. Mixed 0-4 scales enter only through their _applies / _risk columns; blanks that mean
"does not apply" become 0 with the _applies flag carrying the information; the 8 missing ages are filled with the
training-fold median. Also prints the trivial baselines an LLM must beat.

Run from inside jakarta_hiv_project/:  python scripts/baseline_check.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
OUT = os.path.join('outputs', 'baseline_check.txt')

d = pd.read_csv(os.path.join('data', 'processed', 'jakarta_hiv_clean.csv'))
dic = pd.read_csv(os.path.join('data', 'processed', 'jakarta_hiv_data_dictionary.csv')).set_index('column')
d = d[d['igg_borderline'] == 0].reset_index(drop=True)
y = d['igg_positive'].to_numpy()

FEATURES = [c for c in d.columns if dic.loc[c, 'role'] == 'feature']
X = d[FEATURES].copy()
for c in FEATURES:
    if c != 'age':
        X[c] = X[c].fillna(0)            # "does not apply" (follow-ups, _risk) -> 0; the 1 blank answer -> 0 + _applies
for c in ['num_cats', 'freq_tooth_decay', 'freq_tooth_loss_extraction', 'freq_tooth_loss_natural_causes',
          'freq_loose_teeth', 'freq_swollen_gums_or_abscess', 'freq_mouth_ulcer_3_months']:
    X[c] = X[c].clip(upper=10)           # a few implausible counts (21 cats, 40 teeth)

EXPOSURE_KEYS = ['cat', 'raw_', 'grilled', 'soil', 'water', 'veggie', 'fruit', 'handwash', 'feces', 'handling_meat',
                 'eating_with_hands', 'nail']
SETS = {
    'all questions': FEATURES,
    'exposures (cats, food, water, soil, hygiene)': [c for c in FEATURES if any(k in c for k in EXPOSURE_KEYS)],
    'background (age, gender, education, household goods)': ['age', 'gender', 'education', 'cooking_fuel', 'owns_land',
                                                             'electricity_capacity_1300_watt_or_more']
                                                            + [c for c in FEATURES if c.startswith('has_') and c != 'has_cat'],
    'age only': ['age'],
}
MODELS = {
    'LogReg': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                    LogisticRegression(C=0.1, max_iter=5000)),
    'RandomForest': lambda: make_pipeline(SimpleImputer(strategy='median'),
                                          RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0)),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

lines = [f'Jakarta HIV data: {len(d)} people ({y.sum()} positive, {y.mean():.0%}); 5-fold CV x 20 repeats', '']
lines.append(f'{"feature set":52} {"model":12} {"ROC-AUC":>15} {"PR-AUC":>15} {"Brier":>7}')
for name, cols in SETS.items():
    for mname, make in MODELS.items():
        if name == 'age only' and mname == 'RandomForest':
            continue
        s = cross_validate(make(), X[cols], y, cv=cv, n_jobs=-1,
                           scoring={'roc': 'roc_auc', 'pr': 'average_precision', 'brier': 'neg_brier_score'})
        lines.append(f'{name + f" ({len(cols)})":52} {mname:12} {s["test_roc"].mean():.3f} +/- {s["test_roc"].std():.3f}'
                     f'  {s["test_pr"].mean():.3f} +/- {s["test_pr"].std():.3f}  {-s["test_brier"].mean():.3f}')
p = y.mean()
lines += ['', 'Trivial baselines (what an LLM must beat):',
          f'  everyone gets the same risk ({p:.2f}):  ROC-AUC 0.500, PR-AUC {p:.3f}, Brier {p * (1 - p):.3f}',
          f'  everyone called positive:            F1 (positive class) {f1_score(y, np.ones_like(y)):.3f}, '
          f'weighted F1 {f1_score(y, np.ones_like(y), average="weighted"):.3f}, accuracy {p:.3f}']
os.makedirs('outputs', exist_ok=True)
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nSaved {OUT}')

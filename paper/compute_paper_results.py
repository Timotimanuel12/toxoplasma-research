"""Compute the model results used in the paper (standard ML models only). Run from the repo root.
Part 1: four standard models x three feature sets on NHANES 2013-2014 (same split and 15 CV folds as before).
Part 2: the improvement steps of research_r/toxo_max_score.ipynb (re-run from the notebook code) for ROC curves.
Output: paper/results/*.json / *.csv
"""
import json, os, sys, warnings
warnings.filterwarnings('ignore'); os.environ['PYTHONWARNINGS'] = 'ignore'
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import build_bio_features as bbf

OUT = os.path.join('paper', 'results'); os.makedirs(OUT, exist_ok=True)
T, RS = 'target_toxo_pos', 42
df = pd.read_csv(bbf.OUT_CSV)
CAT = ['RIAGENDR', 'RIDRETH3', 'yrs_in_us_grp', 'SLQ050', 'smoking_status', 'toxocara_pos']
DEMO = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'yrs_in_us_grp', 'DMDEDUC2', 'INDFMPIR']
BEH = ['HSD010', 'SLD010H', 'SLQ050', 'smoking_status', 'drinks_per_day', 'phq9_total_score']
SETS = {'Baseline': DEMO + ['toxocara_pos'],
        'Baseline + lifestyle': DEMO + ['toxocara_pos'] + BEH,
        'Baseline + lifestyle + 44 markers': DEMO + ['toxocara_pos'] + BEH + bbf.NEW_FEATURES}
MODELS = {'Logistic Regression': LogisticRegression(C=1.0, max_iter=5000),
          'Random Forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=10, n_jobs=1, random_state=RS),
          'Gradient Boosting': HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=300, random_state=RS),
          'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                   eval_metric='logloss', n_jobs=1, random_state=RS)}


def prep(cols):
    cat = [c for c in cols if c in CAT]; num = [c for c in cols if c not in CAT]
    return ColumnTransformer([('c', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                                              ('o', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))]), cat),
                              ('n', Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True)), ('s', StandardScaler())]), num)])


X, y = df.drop(columns=[T]), df[T]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
CV = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS).split(Xtr, ytr))


def fold(m, s, tr, va):
    cols = SETS[s]
    p = make_pipeline(prep(cols), clone(MODELS[m])).fit(Xtr.iloc[tr][cols], ytr.iloc[tr])
    return m, s, roc_auc_score(ytr.iloc[va], p.predict_proba(Xtr.iloc[va][cols])[:, 1])


def final(m, s):
    cols = SETS[s]
    p = make_pipeline(prep(cols), clone(MODELS[m])).fit(Xtr[cols], ytr)
    return m, s, p.predict_proba(Xte[cols])[:, 1]


cv = Parallel(n_jobs=-1)(delayed(fold)(m, s, tr, va) for m in MODELS for s in SETS for tr, va in CV)
te = Parallel(n_jobs=-1)(delayed(final)(m, s) for m in MODELS for s in SETS)
cvd = pd.DataFrame(cv, columns=['model', 'set', 'auc'])
rows = []
for (m, s), g in cvd.groupby(['model', 'set'], sort=False):
    p = [pp for mm, ss, pp in te if mm == m and ss == s][0]
    rows.append({'model': m, 'set': s, 'cv_mean': g.auc.mean(), 'cv_sd': g.auc.std(), 'test_auc': roc_auc_score(yte, p)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'models_by_featureset.csv'), index=False)
print(res.round(4).to_string(index=False))
curves = {f'{m}|{s}': {'fpr': roc_curve(yte, p)[0].tolist(), 'tpr': roc_curve(yte, p)[1].tolist(), 'auc': roc_auc_score(yte, p)}
          for m, s, p in te}

# ---------------- Part 2: improvement steps (code from toxo_max_score.ipynb, cells 1, 10, 11)
nb = json.load(open(os.path.join('research_r', 'toxo_max_score.ipynb')))
g = {'display': lambda *a, **k: None}
for i in (1, 10, 11):
    exec(''.join(nb['cells'][i]['source']), g)
yt = g['test'][g['TARGET']].to_numpy()
for name in ['0 earlier best (2013-14 only)', '5 + XGBoost ensemble']:
    p = g['test_p'][name]
    curves['STEP|' + name] = {'fpr': roc_curve(yt, p)[0].tolist(), 'tpr': roc_curve(yt, p)[1].tolist(), 'auc': roc_auc_score(yt, p)}
lad = g['ladder']
lad.to_csv(os.path.join(OUT, 'improvement_steps.csv'), index=False)
print(lad.round(4).to_string(index=False))
json.dump(curves, open(os.path.join(OUT, 'roc_curves.json'), 'w'))
print('saved')

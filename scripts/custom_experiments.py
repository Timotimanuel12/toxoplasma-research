"""
Experiments for research_r/toxo_custom_model.ipynb.

Data: NHANES 2009-2014 adults (same Toxoplasma assay), training = everyone except the 1,002 held-out
2013-2014 test people used since the first notebook. CV = 5-fold x 2 repeats on the training people, AUC computed on
the 2013-2014 people inside each validation fold (same population as all earlier notebooks).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
import build_bio_features as bbf  # noqa: E402
from catalytic_models import CatalyticBoost, CatalyticGLM  # noqa: E402

T = 'target_toxo_pos'
SEL = ['hav_natural_infection', 'hav_antibody', 'hev_igg', 'hsv1', 'hsv2', 'toxocara_mfi_log', 'wbc_count',
       'eosinophil_pct', 'lymphocyte_pct', 'blood_lead_log', 'blood_mercury_log', 'cotinine_log',
       'well_or_spring_water', 'no_tap_water', 'household_size', 'crowding', 'home_rented', 'spanish_interview',
       'not_us_citizen', 'work_food_service', 'work_construction_soil', 'food_insecurity', 'no_health_insurance',
       'inject_drugs_ever', 'hav_vaccinated', 'missing_teeth']   # the 26 factors selected on training rows in toxo_max_score


def feature_matrix(df):
    X = pd.DataFrame({'age': df['RIDAGEYR'], 'male': (df['RIAGENDR'] == 1).astype(float)})
    for c in ['RIDRETH1', 'yrs_in_us_grp', 'cycle', 'SLQ050', 'smoking_status']:
        X = X.join(pd.get_dummies(df[c], prefix=c, dtype=float))
    for c in ['DMDEDUC2', 'INDFMPIR', 'toxocara_pos', 'HSD010', 'SLD010H', 'drinks_per_day', 'phq9_total_score'] + SEL:
        X[c] = df[c]
    X['born_abroad'] = (df['DMDBORN4'] == 2).astype(float)
    return X.astype(float)


def load():
    base = pd.read_csv(bbf.BASE_CSV)
    _, bt = train_test_split(base, test_size=0.2, stratify=base[T], random_state=42)
    p = pd.read_csv(bbf.OUT_POOLED_CSV)
    titers = []
    for s in 'FGH':
        t = bbf.read('SSTOXO', s, ['SSTOXG', 'SSTOXO'])
        t['igg'] = t['SSTOXG'].fillna(t['SSTOXO'])
        titers.append(t[['SEQN', 'igg']])
    p = p.merge(pd.concat(titers), on='SEQN', how='left')
    X = feature_matrix(p)
    te = p['SEQN'].isin(bt['SEQN']).to_numpy()
    D = dict(Xtr=X[~te].reset_index(drop=True), ytr=p.loc[~te, T].to_numpy(),
             Htr=(p.loc[~te, 'cycle'] == '2013-2014').to_numpy(), igg_tr=p.loc[~te, 'igg'].to_numpy(),
             Xte=X[te].reset_index(drop=True), yte=p.loc[te, T].to_numpy())
    D['med'] = D['Xtr'].median()
    D['flag_cols'] = [c for c in D['Xtr'].columns if D['Xtr'][c].isna().mean() > 0.02]
    D['CV'] = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(D['Xtr'], D['ytr']))
    return D


# ------------------------------------------------------------------ model wrappers (DataFrame in)
class _Impute:
    def __init__(self, med, flag_cols):
        self.med, self.flag_cols = med, flag_cols

    def __call__(self, A):
        flags = A[self.flag_cols].isna().astype(float).add_suffix('_missing')
        return pd.concat([A.fillna(self.med), flags], axis=1)


class LRSplineInter:
    """Logistic regression, age spline + age x born-abroad / education / income (step 4 of toxo_max_score)."""
    def __init__(self, imp, C=0.3):
        self.imp, self.C = imp, C

    def _z(self, A, fit=False):
        B = self.imp(A)
        if fit:
            self.sp = SplineTransformer(n_knots=6, degree=3).fit(B[['age']])
        Z = np.hstack([B.drop(columns='age').to_numpy(), self.sp.transform(B[['age']]),
                       (B['age'] * B['born_abroad']).to_numpy()[:, None], (B['age'] * B['DMDEDUC2']).to_numpy()[:, None],
                       (B['age'] * B['INDFMPIR']).to_numpy()[:, None]])
        if fit:
            self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        return (Z - self.mu) / self.sd

    def fit(self, A, y):
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self._z(A, True), y)
        return self

    def predict_proba(self, A):
        return self.m.predict_proba(self._z(A))


class GLMWrap:
    def __init__(self, imp, **kw):
        self.imp, self.kw = imp, kw

    def fit(self, A, y):
        self.cols = list(self.imp(A).columns)
        self.m = CatalyticGLM(age_col=self.cols.index('age'), **self.kw).fit(self.imp(A).to_numpy(), y)
        return self

    def predict_proba(self, A):
        return self.m.predict_proba(self.imp(A).to_numpy())


class BoostWrap:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, A, y):
        self.m = CatalyticBoost(age_col=list(A.columns).index('age'), **self.kw).fit(A.to_numpy(), y)
        return self

    def predict_proba(self, A):
        return self.m.predict_proba(A.to_numpy())


class XGBWrap:
    def fit(self, A, y):
        self.m = XGBClassifier(n_estimators=800, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.6,
                               min_child_weight=10, reg_lambda=5, eval_metric='logloss', n_jobs=1, random_state=0).fit(A, y)
        return self

    def predict_proba(self, A):
        return self.m.predict_proba(A)


class SmallLR:
    def fit(self, A, y):
        self.m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                               LogisticRegression(C=0.3, max_iter=5000)).fit(A, y)
        return self

    def predict_proba(self, A):
        return self.m.predict_proba(A)


class Average:
    def __init__(self, makers):
        self.makers = makers

    def fit(self, A, y):
        self.ms = [mk().fit(A, y) for mk in self.makers]
        return self

    def predict_proba(self, A):
        p = np.mean([m.predict_proba(A)[:, 1] for m in self.ms], axis=0)
        return np.column_stack([1 - p, p])


def model_makers(D):
    imp = _Impute(D['med'], D['flag_cols'])
    lr = lambda: LRSplineInter(imp)
    glm = lambda: GLMWrap(imp, alpha=50.0)
    cb = lambda: BoostWrap()
    xg = lambda: XGBWrap()
    return {
        'Logistic regression (spline + interactions)': lr,
        'XGBoost (standard)': xg,
        'CatalyticGLM (custom)': glm,
        'CatalyticBoost (custom)': cb,
        'Ensemble: logistic + CatalyticBoost': lambda: Average([lr, cb]),
        'Ensemble: logistic + XGBoost + CatalyticGLM + CatalyticBoost': lambda: Average([lr, xg, glm, cb]),
    }


def _cv_job(D, mk, tr, va, fit_mask, pop):
    trm = tr[pop[tr] & fit_mask[tr]]
    cols = D['cols']
    m = mk().fit(D['Xtr'].iloc[trm][cols], D['ytr'][trm])
    vh = va[pop[va] & D['Htr'][va]]
    return roc_auc_score(D['ytr'][vh], m.predict_proba(D['Xtr'].iloc[vh][cols])[:, 1])


def evaluate(D, mk, cols=None, fit_mask=None, pop_tr=None, pop_te=None, n_jobs=-1):
    """CV AUC (2013-14 people) and test AUC. fit_mask: rows allowed for FITTING only (label cleaning).
    pop_tr/pop_te: population restriction applied to fitting AND scoring (pre-specified subgroup)."""
    n = len(D['ytr'])
    D = dict(D, cols=cols or list(D['Xtr'].columns))
    fit_mask = np.ones(n, bool) if fit_mask is None else fit_mask
    pop_tr = np.ones(n, bool) if pop_tr is None else pop_tr
    pop_te = np.ones(len(D['yte']), bool) if pop_te is None else pop_te
    cv = Parallel(n_jobs=n_jobs)(delayed(_cv_job)(D, mk, tr, va, fit_mask, pop_tr) for tr, va in D['CV'])
    rows = np.where(pop_tr & fit_mask)[0]
    m = mk().fit(D['Xtr'].iloc[rows][D['cols']], D['ytr'][rows])
    p = m.predict_proba(D['Xte'][pop_te][D['cols']])[:, 1]
    return {'CV AUC (2013-14 people)': float(np.mean(cv)), 'CV sd': float(np.std(cv)),
            'TEST AUC': roc_auc_score(D['yte'][pop_te], p), 'people used to fit': int(len(rows)),
            'test people': int(pop_te.sum())}, m, p

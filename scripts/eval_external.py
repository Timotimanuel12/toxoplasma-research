"""Repeated cross-validated ROC-AUC for every external dataset (no test-set reuse, fixed model settings)."""
import sys, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import external_datasets as ed
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

N_REPEATS = 20
MODELS = {
    'LogReg': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
    'RandomForest': lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=1, random_state=0)),
    'XGBoost': lambda: XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, eval_metric='logloss', n_jobs=1, random_state=0),
}

def run(name, X, y, model):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=42)
    oof = np.zeros((N_REPEATS, len(y)))
    for i, (tr, va) in enumerate(rskf.split(X, y)):
        m = MODELS[model]().fit(X.iloc[tr], y.iloc[tr])
        oof[i // 5, va] = m.predict_proba(X.iloc[va])[:, 1]
    aucs = np.array([roc_auc_score(y, o) for o in oof])
    return name, model, aucs

if __name__ == '__main__':
    data = ed.load_all()
    jobs = [(k, X, y, m) for k, (X, y, _) in data.items() for m in MODELS]
    out = Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs)
    rows = []
    for k, m, a in out:
        X, y, info = data[k]
        rows.append({'dataset': k, 'setting': info['setting'], 'n': len(y), 'positive %': round(100 * y.mean(), 1),
                     'model': m, 'CV AUC': a.mean(), 'low (2.5%)': np.percentile(a, 2.5), 'high (97.5%)': np.percentile(a, 97.5)})
    res = pd.DataFrame(rows)
    res.to_csv('data/processed/external_cv_auc.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.pivot_table(index=['dataset', 'n', 'positive %'], columns='model', values='CV AUC').round(3).to_string())

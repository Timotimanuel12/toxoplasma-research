"""
'Train harder' experiment: (1) heavy Optuna hyperparameter search for XGBoost and LightGBM,
(2) what happens when the model is trained harder (more/deeper trees) -> train vs validation AUC.
Same data as toxo_max_score.ipynb step 5: pooled 2009-2014 training people, the same 1,002 test people.
"""
import sys, os, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import build_bio_features as bbf
import optuna, lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)

T = 'target_toxo_pos'
base = pd.read_csv(bbf.BASE_CSV)
_, bt = train_test_split(base, test_size=0.2, stratify=base[T], random_state=42)
p = pd.read_csv(bbf.OUT_POOLED_CSV)
p['born_abroad'] = (p['DMDBORN4'] == 2).astype(float)
p['cycle_n'] = p['cycle'].map({'2009-2010': 0, '2011-2012': 1, '2013-2014': 2})
for c in ['RIAGENDR', 'RIDRETH1', 'yrs_in_us_grp', 'SLQ050', 'smoking_status']:
    p = pd.concat([p, pd.get_dummies(p[c], prefix=c, dtype=float)], axis=1)
dummies = [c for c in p.columns if any(c.startswith(x + '_') for x in ['RIAGENDR', 'RIDRETH1', 'yrs_in_us_grp', 'SLQ050', 'smoking_status']) and c not in ('yrs_in_us_grp',)]
FEATS = dummies + ['RIDAGEYR', 'DMDEDUC2', 'INDFMPIR', 'HSD010', 'SLD010H', 'drinks_per_day', 'phq9_total_score',
                   'toxocara_pos', 'cycle_n', 'born_abroad'] + bbf.NEW_FEATURES
FEATS = list(dict.fromkeys(FEATS))
train = p[~p.SEQN.isin(bt.SEQN)].reset_index(drop=True)
test = p[p.SEQN.isin(bt.SEQN)].reset_index(drop=True)
X, y = train[FEATS].astype(float), train[T].to_numpy()
Xt, yt = test[FEATS].astype(float), test[T].to_numpy()
is_h = (train.cycle == '2013-2014').to_numpy()
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
print(f'{len(FEATS)} features, train {len(X)}, test {len(Xt)}')

def cv_score(make):
    oof = np.zeros(len(X)); tr_auc = []
    for tr, va in folds:
        m = make().fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        tr_auc.append(roc_auc_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
    return roc_auc_score(y[is_h], oof[is_h]), np.mean(tr_auc)

out = {}
# ---------------- 1. train harder: bigger / deeper models ----------------
rows = []
for depth, n in [(2, 300), (3, 800), (4, 1500), (6, 2000), (8, 3000), (10, 5000)]:
    mk = lambda: XGBClassifier(n_estimators=n, learning_rate=0.02, max_depth=depth, subsample=0.8,
                               colsample_bytree=0.6, eval_metric='logloss', n_jobs=2, random_state=42)
    va, tr = cv_score(mk)
    rows.append({'max_depth': depth, 'trees': n, 'train AUC': tr, 'validation AUC (2013-14)': va})
    print(rows[-1])
out['harder'] = rows

# ---------------- 2. Optuna search ----------------
def xgb_obj(trial):
    prm = dict(n_estimators=trial.suggest_int('n_estimators', 200, 2500, log=True),
               learning_rate=trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
               max_depth=trial.suggest_int('max_depth', 1, 7),
               min_child_weight=trial.suggest_float('min_child_weight', 1, 50, log=True),
               subsample=trial.suggest_float('subsample', 0.5, 1.0),
               colsample_bytree=trial.suggest_float('colsample_bytree', 0.2, 1.0),
               reg_lambda=trial.suggest_float('reg_lambda', 0.01, 50, log=True),
               reg_alpha=trial.suggest_float('reg_alpha', 0.001, 10, log=True),
               gamma=trial.suggest_float('gamma', 0, 5))
    return cv_score(lambda: XGBClassifier(**prm, eval_metric='logloss', n_jobs=2, random_state=42))[0]

def lgb_obj(trial):
    prm = dict(n_estimators=trial.suggest_int('n_estimators', 200, 2500, log=True),
               learning_rate=trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
               num_leaves=trial.suggest_int('num_leaves', 2, 64, log=True),
               min_child_samples=trial.suggest_int('min_child_samples', 10, 300, log=True),
               subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
               colsample_bytree=trial.suggest_float('colsample_bytree', 0.2, 1.0),
               reg_lambda=trial.suggest_float('reg_lambda', 0.01, 50, log=True))
    return cv_score(lambda: lgb.LGBMClassifier(**prm, verbose=-1, n_jobs=2, random_state=42))[0]

budget = int(os.environ.get('TRIALS', 60))
for name, obj, cls in [('XGBoost', xgb_obj, 'xgb'), ('LightGBM', lgb_obj, 'lgb')]:
    t = time.time()
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, n_trials=budget)
    bp = st.best_params
    m = (XGBClassifier(**bp, eval_metric='logloss', n_jobs=2, random_state=42) if cls == 'xgb'
         else lgb.LGBMClassifier(**bp, subsample_freq=1, verbose=-1, n_jobs=2, random_state=42)).fit(X, y)
    test_auc = roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
    out[name] = {'trials': budget, 'best CV AUC (2013-14)': st.best_value, 'test AUC': test_auc, 'params': bp,
                 'all trial CV AUCs': [tr.value for tr in st.trials]}
    print(f'{name}: {budget} trials, best CV {st.best_value:.4f}, test {test_auc:.4f}, {time.time()-t:.0f}s')
json.dump(out, open('data/processed/train_harder_results.json', 'w'), indent=1, default=float)

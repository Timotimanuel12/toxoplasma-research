"""
Combine US (NHANES 2009-2014, NHANES III) with Europe (Czech/Slovak, France) and test whether the extra
European people improve prediction for US people.

Every model is scored on the SAME 1,002 NHANES 2013-2014 test people used since the first notebook.
Pooled features: age and sex are shared by every dataset; every other question keeps its own column
(missing where a dataset did not ask it) plus a dataset indicator, so XGBoost can use what each source has.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import build_bio_features as bbf
import build_nhanes3 as b3
import europe_datasets as eu

T = 'target_toxo_pos'
US_FEATS = ['RIDRETH1', 'yrs_in_us_grp', 'DMDEDUC2', 'INDFMPIR', 'toxocara_pos'] + bbf.NEW_FEATURES


def load():
    base = pd.read_csv(bbf.BASE_CSV)
    _, bt = train_test_split(base, test_size=0.2, stratify=base[T], random_state=42)
    p = pd.read_csv(bbf.OUT_POOLED_CSV)
    us = pd.DataFrame({'age': p['RIDAGEYR'], 'male': (p['RIAGENDR'] == 1).astype(float)})
    for c in US_FEATS:
        us['us_' + c] = p[c]
    us['y'] = p[T]; us['dataset'] = 'US NHANES 2009-14'; us['seqn'] = p['SEQN']
    test = us[us['seqn'].isin(bt['SEQN'])].reset_index(drop=True)
    us_train = us[~us['seqn'].isin(bt['SEQN'])].reset_index(drop=True)

    n3 = pd.read_csv(b3.OUT_CSV)
    u3 = pd.DataFrame({'age': n3['age'], 'male': n3['male']})
    for c in b3.EXPOSURE3 + b3.LAB3 + ['race', 'birthplace', 'education_years', 'poverty_income_ratio']:
        u3['n3_' + c] = n3[c]
    u3['y'] = n3[T]; u3['dataset'] = 'US NHANES III 1988-94'

    frames = {}
    for name, f in [('Czech/Slovak', eu.czech_slovak), ('France donors', eu.france_donors)]:
        X, y = f()
        e = pd.DataFrame({'age': X['age'], 'male': X['male']})
        for c in X.columns.drop(['age', 'male']):
            e[('cz_' if name.startswith('Czech') else 'fr_') + c] = X[c]
        e['y'] = y.values; e['dataset'] = name
        frames[name] = e
    return us_train, test, u3, frames


def xgb():
    return XGBClassifier(n_estimators=800, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.6,
                         min_child_weight=10, reg_lambda=5, eval_metric='logloss', n_jobs=-1, random_state=42)


def design(df, cols):
    X = df.reindex(columns=cols).astype(float)
    for d in ['US NHANES 2009-14', 'US NHANES III 1988-94', 'Czech/Slovak', 'France donors']:
        X['is_' + d] = (df['dataset'] == d).astype(float)
    return X


def main():
    us_train, test, u3, eu_frames = load()
    combos = {
        'US 2009-14 only': [us_train],
        '+ NHANES III (US)': [us_train, u3],
        '+ Europe (Czech/Slovak + France)': [us_train] + list(eu_frames.values()),
        '+ NHANES III + Europe (everything)': [us_train, u3] + list(eu_frames.values()),
    }
    rows = []
    for label, parts in combos.items():
        tr = pd.concat(parts, ignore_index=True)
        cols = [c for c in tr.columns if c not in ('y', 'dataset', 'seqn')]
        aucs = []
        for seed in range(5):
            m = xgb().set_params(random_state=seed).fit(design(tr, cols), tr['y'])
            aucs.append(roc_auc_score(test['y'], m.predict_proba(design(test, cols))[:, 1]))
        rows.append({'training data': label, 'training people': len(tr),
                     'US test AUC (1,002 NHANES people)': np.mean(aucs), 'sd over 5 seeds': np.std(aucs)})
    res = pd.DataFrame(rows)

    # pooled cross-validation over everything: the misleading 'combined' number
    allp = pd.concat([us_train, test, u3] + list(eu_frames.values()), ignore_index=True)
    cols = [c for c in allp.columns if c not in ('y', 'dataset', 'seqn')]
    oof = np.zeros(len(allp))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(allp, allp['y']):
        m = xgb().fit(design(allp.iloc[tr], cols), allp['y'].iloc[tr])
        oof[va] = m.predict_proba(design(allp.iloc[va], cols))[:, 1]
    within = allp.assign(p=oof).groupby('dataset').apply(lambda g: pd.Series({
        'people': len(g), 'positive %': round(100 * g['y'].mean(), 1), 'AUC within this dataset': roc_auc_score(g['y'], g['p'])}))
    pooled_auc = roc_auc_score(allp['y'], oof)
    return res, within, pooled_auc


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    res, within, pooled = main()
    print(res.round(4).to_string(index=False))
    print(f'\nPooled 5-fold CV AUC over all {int(within.people.sum()):,} people (all countries mixed): {pooled:.3f}')
    print(within.round(3).to_string())

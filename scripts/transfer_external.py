"""Leave-one-study-out: train on 8 studies with harmonised questions, test on the 9th."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
import external_datasets as ed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MAP = {  # harmonised question -> column in each study (None = not asked)
    'cat':   {'indonesia_pregnant': 'cat_contact', 'cameroon_hiv': 'cat_owner', 'china_women': 'cat_contact',
              'iran_suicide': 'cat_contact_code', 'philippines_cebu': 'cat_owner', 'ghana_sickle_cell': 'cat_owner',
              'brazil_homeless': 'Cat owner'},
    'raw_or_undercooked_meat': {'indonesia_pregnant': 'eats_chicken_satay', 'cameroon_hiv': 'raw_meat', 'china_women': 'raw_meat',
              'myanmar_slaughter': 'raw_meat', 'iran_suicide': 'raw_meat_code', 'ghana_sickle_cell': 'undercooked_meat'},
    'soil':  {'indonesia_pregnant': 'soil_contact', 'cameroon_hiv': 'soil_contact', 'china_women': 'soil_contact',
              'myanmar_slaughter': 'soil_contact'},
    'raw_vegetables': {'indonesia_pregnant': 'raw_vegetables', 'cameroon_hiv': 'raw_vegetables', 'china_women': 'raw_vegetables'},
    'untreated_water': {'indonesia_pregnant': 'unboiled_water', 'cameroon_hiv': 'water_untreated'},
    'age': {k: 'age' for k in ['indonesia_pregnant', 'cameroon_hiv', 'china_women', 'myanmar_slaughter', 'philippines_cebu',
                               'thai_border_pregnant', 'brazil_homeless', 'ghana_sickle_cell']},
}
RECODE12 = {'cat_contact_code', 'raw_meat_code', 'Cat owner'}     # 1 = yes, 2 = no in these files


def harmonise(data):
    frames = []
    for k, (X, y, _) in data.items():
        h = pd.DataFrame(index=X.index)
        for q, m in MAP.items():
            col = m.get(k)
            v = X[col] if col else pd.Series(np.nan, index=X.index)
            if col in RECODE12:
                v = v.map({1: 1.0, 2: 0.0})
            h[q] = v
        h['age'] = h['age'] / 10
        h['study'] = k
        h['y'] = y.values
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def design(df):
    X = pd.DataFrame(index=df.index)
    for q in MAP:
        X[q] = df[q].fillna(0.0)
        X[q + '_asked'] = df[q].notna().astype(float)
    return X


if __name__ == '__main__':
    H = harmonise(ed.load_all())
    rows = []
    for s in H['study'].unique():
        tr, te = H[H.study != s], H[H.study == s]
        m = LogisticRegression(C=1.0, max_iter=5000).fit(design(tr), tr['y'])
        asked = [q for q in MAP if te[q].notna().any()]
        auc = roc_auc_score(te['y'], m.predict_proba(design(te))[:, 1]) if asked else np.nan
        rows.append({'held-out study': s, 'n': len(te), 'questions available': ', '.join(asked), 'transfer AUC': round(auc, 3)})
    res = pd.DataFrame(rows)
    res.to_csv('data/processed/external_transfer_auc.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_colwidth', 80)
    print(res.to_string(index=False))
    coef = LogisticRegression(C=1.0, max_iter=5000).fit(design(H), H['y'])
    print(pd.Series(np.exp(coef.coef_[0]), index=design(H).columns).round(2).to_string())

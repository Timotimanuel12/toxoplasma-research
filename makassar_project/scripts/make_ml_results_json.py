"""
Logistic-regression results for the 182 Makassar women, in the SAME JSON-lines format as the teammate's LLM results
(pid, true_label, answer{risk_score, risk_factors, explanation, label}, error, raw), so ML and LLM can be compared
line by line. Every record says source = "logistic regression (ML baseline)": this is NOT LLM output.

  risk_score   : out-of-fold risk x 100 (5-fold CV x 20 repeats, averaged; each woman scored only by models that
                 never saw her) - the same numbers as ml_predictions.csv (lr_all)
  risk_factors : the 5 answers that move THIS woman's risk most, from a model fitted on all 182 women
                 (contribution = coefficient x her standardised answer); direction from its sign
  explanation  : a fixed template filled with those factors (not written by an AI)
  label        : 1 if risk_score >= 50

Run from inside makassar_project/:  python scripts/make_ml_results_json.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURE_SETS, load_data, make_cv_splits, make_preprocessor

OUT = os.path.join('data', 'jsonllm', 'makassar_ml_logreg_182.jsonl')
FEATURES = FEATURE_SETS['C_all']
WATER = {'pipe': 'piped (tap) water', 'pump': 'pumped well water', 'surface': 'surface water',
         'gallon': 'refill-gallon water'}
EDU = {1: 'elementary school', 2: 'junior high school', 3: 'senior high school', 4: 'above senior high school'}
yn = lambda v, yes, no: yes if v == 1 else no
DESCRIBE = {   # how each answer is written as a factor, from the woman's own answer
    'age': lambda r: f'aged {int(r.age)}',
    'trimester': lambda r: f'trimester {int(r.trimester)}',
    'gravidity': lambda r: f'pregnancy number {int(r.gravidity)}',
    'parity': lambda r: f'{int(r.parity)} previous births',
    'education': lambda r: f'education: {EDU[int(r.education)]}',
    'abortion_history': lambda r: yn(r.abortion_history, 'previous miscarriage', 'no previous miscarriage'),
    'employed': lambda r: yn(r.employed, 'employed', 'not employed'),
    'raises_cat': lambda r: yn(r.raises_cat, 'raises a cat', 'does not raise a cat'),
    'cat_contact': lambda r: yn(r.cat_contact, 'has contact with cats', 'no contact with cats'),
    'cleans_litter': lambda r: yn(r.cleans_litter, 'cleans cat litter', 'does not clean cat litter'),
    'soil_contact': lambda r: yn(r.soil_contact, 'has contact with soil', 'no contact with soil'),
    'eats_chicken_satay': lambda r: yn(r.eats_chicken_satay, 'ate undercooked chicken satay', 'did not eat undercooked chicken satay'),
    'eats_raw_vegetables': lambda r: yn(r.eats_raw_vegetables, 'ate raw vegetables', 'did not eat raw vegetables'),
    'drinks_unboiled_water': lambda r: yn(r.drinks_unboiled_water, 'drank un-boiled water', 'did not drink un-boiled water'),
    'drinks_gallon_water': lambda r: yn(r.drinks_gallon_water, 'drinks filtered refill-gallon water', 'drinks boiled water'),
    'washing_water': lambda r: f'washing water: {WATER[r.washing_water]}',
    'cooking_water': lambda r: f'cooking water: {WATER[r.cooking_water]}',
}
assert set(DESCRIBE) == set(FEATURES)

df = load_data()
y = df['igg_positive'].to_numpy()
X = df[FEATURES]


def model():
    return Pipeline([('prepare', make_preprocessor(FEATURES)), ('model', LogisticRegression(C=1.0, max_iter=5000))])


# 1. out-of-fold risk (identical procedure to make_llm_handoff.py)
splits = make_cv_splits(X, y, n_repeats=20, seed=42)
risk = np.zeros(len(df))
for tr, va in splits:
    risk[va] += model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1] / 20
check = pd.read_csv(os.path.join('outputs', 'llm_handoff', 'ml_predictions.csv'))
assert np.allclose(check['lr_all'].to_numpy(), risk.round(4), atol=1e-4), 'differs from ml_predictions.csv'

# 2. per-woman contributions from a model fitted on everyone (one-hot water columns summed back per question)
full = model().fit(X, y)
Z = full.named_steps['prepare'].transform(X)
names = full.named_steps['prepare'].get_feature_names_out()
contrib = pd.DataFrame(Z * full.named_steps['model'].coef_[0], columns=names)
orig = [n.split('__', 1)[1] for n in names]
orig = [next(f for f in FEATURES if o == f or o.startswith(f + '_')) for o in orig]
contrib = contrib.T.groupby(orig).sum().T

with open(OUT, 'w', encoding='utf-8') as f:
    for i, r in df.iterrows():
        top = contrib.loc[i].reindex(contrib.loc[i].abs().sort_values(ascending=False).index)[:5]
        factors = [{'factor': DESCRIBE[k](r), 'direction': 'increases' if v > 0 else 'decreases'} for k, v in top.items()]
        score = int(round(risk[i] * 100))
        up = [x['factor'] for x in factors if x['direction'] == 'increases']
        down = [x['factor'] for x in factors if x['direction'] == 'decreases']
        expl = (f'Logistic regression trained on the other women gives a risk of {score}%. '
                + (f'Raising her risk: {", ".join(up)}. ' if up else '')
                + (f'Lowering her risk: {", ".join(down)}.' if down else '')).strip()
        answer = {'risk_score': score, 'risk_factors': factors, 'explanation': expl, 'label': int(score >= 50)}
        f.write(json.dumps({'pid': r['id'], 'true_label': int(y[i]), 'source': 'logistic regression (ML baseline)',
                            'answer': answer, 'error': None, 'raw': json.dumps(answer)}, ensure_ascii=False) + '\n')

print(f'Saved {OUT}: {len(df)} women; ROC-AUC of risk_score {roc_auc_score(y, risk):.3f}; '
      f'labelled positive {int((risk >= 0.5).sum())} (true positives in data {y.sum()})')
print('Model weights (standardised, positive = raises risk):')
w = pd.Series(full.named_steps['model'].coef_[0], index=names).sort_values()
print('  ' + ', '.join(f'{n.split("__")[1]} {v:+.2f}' for n, v in w.items()))

"""
Package the Makassar data for the LLM screening test (teammate's RQ4 part).

Writes to outputs/llm_handoff/:
  profiles.csv        id, site, profile  -> one English text profile per woman, answers only (no blood test)
  labels.csv          id, igg_positive   -> the answer key; open only after the LLM has scored every profile
  ml_predictions.csv  id, lr_all, lr_exposures -> out-of-fold risk from our logistic regression (same 20x5 folds
                      as train_baselines.py), so the LLM can be compared with ML on the same women
  makassar_llm_text.csv  row_id, text, serostatus -> the same profiles in the teammate's Jakarta input format
  row_id_to_woman_id.csv row_id, id                -> which woman each row_id is

Same 182 women as all other Makassar experiments (2 borderline IgG results excluded).
Run from inside makassar_project/:  python scripts/make_llm_handoff.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEATURE_SETS, OUT_DIR, load_data, make_cv_splits, make_preprocessor

HANDOFF_DIR = os.path.join(OUT_DIR, 'llm_handoff')

TRIMESTER = {1: 'first', 2: 'second', 3: 'third'}
EDUCATION = {1: 'elementary school', 2: 'junior high school', 3: 'senior high school',
             4: 'education above senior high school'}
WATER = {'pipe': 'piped (tap) water', 'pump': 'pumped well water', 'surface': 'surface water',
         'gallon': 'refill-gallon water'}


def yes_no(flag, yes, no):
    return yes if flag == 1 else no


def make_profile(r):
    """Fixed template: every question is always mentioned, in the same order, including the 'no' answers."""
    g, p = int(r.gravidity), int(r.parity)
    pregnancy = 'This is her first pregnancy' if g == 1 else f'This is pregnancy number {g} for her'
    births = 'she has no previous births' if p == 0 else f'she has given birth {p} time{"s" if p > 1 else ""} before'
    if r.washing_water == r.cooking_water:
        water = f'Her household uses {WATER[r.washing_water]} for washing and cooking'
    else:
        water = (f'Her household uses {WATER[r.washing_water]} for washing and '
                 f'{WATER[r.cooking_water]} for cooking')
    return ' '.join([
        f'A {int(r.age)}-year-old pregnant woman from Makassar, Indonesia, in her {TRIMESTER[int(r.trimester)]} trimester.',
        f'{pregnancy}; {births}, and {yes_no(r.abortion_history, "she has had a miscarriage before", "she has no history of miscarriage")}.',
        f'She finished {EDUCATION[int(r.education)]} and {yes_no(r.employed, "is employed", "is not employed")}.',
        f'{water}, and she drinks {yes_no(r.drinks_gallon_water, "filtered refill-gallon water", "boiled water")}.',
        f'She {yes_no(r.raises_cat, "raises a cat", "does not raise a cat")},',
        f'{yes_no(r.cat_contact, "has contact with cats", "has no contact with cats")}',
        f'and {yes_no(r.cleans_litter, "cleans cat litter", "does not clean cat litter")}.',
        f'She {yes_no(r.soil_contact, "has contact with soil", "has no contact with soil")}.',
        f'During pregnancy she {yes_no(r.eats_chicken_satay, "ate undercooked chicken satay", "did not eat undercooked chicken satay")},',
        f'{yes_no(r.eats_raw_vegetables, "ate raw vegetables", "did not eat raw vegetables")}',
        f'and {yes_no(r.drinks_unboiled_water, "drank un-boiled water", "did not drink un-boiled water")}.',
    ])


# Phrasing of the teammate's Jakarta input (toxo_llm_text.csv): "Section: answer; answer; ...", one section per line
EDU_JAK = {1: 'finished primary school', 2: 'finished junior high school', 3: 'finished senior high school',
           4: 'studied beyond senior high school'}
WATER_JAK = {'pipe': 'piped tap water (PAM)', 'pump': 'pumped well water', 'surface': 'river or other surface water',
             'gallon': 'refilled gallon water'}


def make_text_profile(r):
    """Same answers as make_profile, in the section style and wording of the teammate's Jakarta profiles."""
    g, p = int(r.gravidity), int(r.parity)
    births = 'no previous births' if p == 0 else f'{p} previous birth{"s" if p > 1 else ""}'
    return '\n'.join([
        f'Demographics: {int(r.age)} years old; female; pregnant, in the {TRIMESTER[int(r.trimester)]} trimester; '
        f'lives in Makassar, Indonesia; education: {EDU_JAK[int(r.education)]}; '
        f'{yes_no(r.employed, "is employed", "is not employed")}.',
        f'Pregnancy history: this is pregnancy number {g}; {births}; '
        f'{yes_no(r.abortion_history, "has had a miscarriage", "has never had a miscarriage")}.',
        f'Cats: {yes_no(r.raises_cat, "keeps a cat", "does not keep a cat")}; '
        f'{yes_no(r.cat_contact, "has contact with cats", "has no contact with cats")}; '
        f'{yes_no(r.cleans_litter, "cleans cat litter", "does not clean cat litter")}.',
        f'Food (during pregnancy): {yes_no(r.eats_chicken_satay, "ate undercooked chicken satay", "did not eat undercooked chicken satay")}; '
        f'{yes_no(r.eats_raw_vegetables, "ate raw vegetables", "did not eat raw vegetables")}.',
        f'Water and soil: {yes_no(r.drinks_gallon_water, "drinks filtered refilled gallon water", "drinks boiled water")}; '
        f'{yes_no(r.drinks_unboiled_water, "drank un-boiled water during pregnancy", "did not drink un-boiled water during pregnancy")}; '
        f'uses {WATER_JAK[r.washing_water]} for bathing and washing; uses {WATER_JAK[r.cooking_water]} for cooking; '
        f'{yes_no(r.soil_contact, "touches soil", "never touches soil")}.',
    ])


def oof_risk(df, features, splits, n_repeats):
    """Out-of-fold predicted risk, averaged over the repeats: each woman is scored only by models that never saw her."""
    X, y = df[features], df['igg_positive'].to_numpy()
    risk = np.zeros(len(df))
    for tr, va in splits:
        model = Pipeline([('prepare', make_preprocessor(features)), ('model', LogisticRegression(C=1.0, max_iter=5000))])
        model.fit(X.iloc[tr], y[tr])
        risk[va] += model.predict_proba(X.iloc[va])[:, 1] / n_repeats
    return risk


df = load_data()
assert len(df) == 182 and df['igg_positive'].sum() == 60
os.makedirs(HANDOFF_DIR, exist_ok=True)

profiles = pd.DataFrame({'id': df['id'], 'site': df['site'], 'profile': df.apply(make_profile, axis=1)})
assert not profiles['profile'].str.contains('IgG|Toxoplasma|positive|negative', case=False).any()
profiles.to_csv(os.path.join(HANDOFF_DIR, 'profiles.csv'), index=False)
df[['id', 'igg_positive']].to_csv(os.path.join(HANDOFF_DIR, 'labels.csv'), index=False)

# Same answers in the teammate's Jakarta input format (toxo_llm_text.csv): row_id, text, serostatus.
# row_id = 0-based row of data/processed/makassar_clean.csv (184 rows; the 2 borderline rows are skipped, as his
# file skips the Jakarta borderline row 129). serostatus is the answer key: it must never go into the prompt.
full = pd.read_csv(os.path.join('data', 'processed', 'makassar_clean.csv'))
row_of = {pid: i for i, pid in enumerate(full['id'])}
txt = df.apply(make_text_profile, axis=1)
assert not txt.str.contains('IgG|Toxoplasma|positive|negative', case=False).any()
llm_in = pd.DataFrame({'row_id': df['id'].map(row_of), 'text': txt, 'serostatus': df['igg_positive']})
assert llm_in['row_id'].is_unique and (full.loc[llm_in['row_id'], 'id'].to_numpy() == df['id'].to_numpy()).all()
llm_in.to_csv(os.path.join(HANDOFF_DIR, 'makassar_llm_text.csv'), index=False)
llm_in.assign(id=df['id'])[['row_id', 'id']].to_csv(os.path.join(HANDOFF_DIR, 'row_id_to_woman_id.csv'), index=False)
old_list = os.path.join(HANDOFF_DIR, 'profiles_list_style.jsonl')   # earlier guess at the format, replaced by this
if os.path.exists(old_list):
    os.remove(old_list)

N_REPEATS = 20
splits = make_cv_splits(df[FEATURE_SETS['C_all']], df['igg_positive'], n_repeats=N_REPEATS, seed=42)
preds = pd.DataFrame({'id': df['id'],
                      'lr_all': oof_risk(df, FEATURE_SETS['C_all'], splits, N_REPEATS).round(4),
                      'lr_exposures': oof_risk(df, FEATURE_SETS['B_exposures'], splits, N_REPEATS).round(4)})
preds.to_csv(os.path.join(HANDOFF_DIR, 'ml_predictions.csv'), index=False)

y = df['igg_positive'].to_numpy()
print(f'Women: {len(df)} ({y.sum()} positive)')
for col in ['lr_all', 'lr_exposures']:
    print(f'{col:13} pooled out-of-fold ROC-AUC {roc_auc_score(y, preds[col]):.3f}  '
          f'PR-AUC {average_precision_score(y, preds[col]):.3f}  Brier {brier_score_loss(y, preds[col]):.3f}')
print('\nExample profile (' + profiles['id'][2] + '):\n' + profiles['profile'][2])
print(f'\nSame woman in the Jakarta input format (makassar_llm_text.csv, row_id {llm_in.row_id[2]}):\n' + txt[2])
print(f'\nProfile length: {profiles["profile"].str.split().str.len().min()}-{profiles["profile"].str.split().str.len().max()} words')
print('Saved to', HANDOFF_DIR)

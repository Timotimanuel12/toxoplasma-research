"""
Clean the Jakarta HIV questionnaire data (186 people living with HIV + 1 borderline; the data behind Ichwan et al.).

Source of truth: data/raw/Toxo sheets - Raw Data.csv (the Google Forms export, answer TEXT).
Reference only:  data/raw/Toxo sheets - cleanedData.csv (an earlier cleaned copy, answer CODES, different row order).
Questionnaire:   docs/questionnaire_Rizky_Fajar_Meirawan.pdf (defines the codes).

Steps (each one checks itself and stops on a failure):
  1. Learn answer text -> code from the people who pair 1-to-1 between the two files.
  2. Convert all raw rows to codes and check that every row equals a row of the earlier cleaned copy.
  3. Recompute age at the interview date (the earlier copy used ~October 2025, and gave 36 to impossible birth dates).
  4. Restructure: id, target, borderline flag, -1 -> missing, mixed 0-4 scales split into applies / risk.
  5. Save data/processed/jakarta_hiv_clean.csv and jakarta_hiv_data_dictionary.csv. Raw files are not changed.

Personal data: date of birth, interview timestamp and the consent answer are used only inside this script and are
not written out. Run from inside jakarta_hiv_project/:  python scripts/clean_jakarta_hiv.py
"""
import os

import numpy as np
import pandas as pd

RAW = os.path.join('data', 'raw', 'Toxo sheets - Raw Data.csv')
OLD = os.path.join('data', 'raw', 'Toxo sheets - cleanedData.csv')
OUT = os.path.join('data', 'processed', 'jakarta_hiv_clean.csv')
DICT = os.path.join('data', 'processed', 'jakarta_hiv_data_dictionary.csv')

# Mixed scales (questionnaire): code 0 = the question does not apply; otherwise 1 = safest ... 4 = riskiest
MIXED_UP = ['oral_condom_use_freq_past_year', 'partner_oral_condom_use_freq_past_year', 'anal_condom_use_past_year',
            'partner_anal_condom_use_past_year', 'freq_handwash_before_after_preparing_food',
            'freq_gloves_use_while_soil_contact_activity_past_year', 'freq_handwash_after_soil_contact_past_year']
# Mixed scales: code 0 = does not eat it raw; otherwise 4 = always washes ... 1 = never washes
MIXED_DOWN = ['freq_veggie_washing_before_consumption', 'freq_fruit_washing_before_consumption']
# Follow-up questions where the earlier copy wrote -1 for "skipped, not applicable"
FOLLOW_UP = ['currently_using_injection_drugs', 'history_of_using_used_needle', 'tattoo_needle_sterile',
             'tattoo_removal_procedure', 'tattoo_removal_sterile', 'cat_to_vet_for_toxoplasma_gondii_check']
TEXT = ['job', 'healthcare_access']
# Codes written out from the questionnaire, for answers that no 1-to-1 pair shows (checked against the learned codes)
EXPLICIT = {'education': {'Tidak Sekolah': 1, 'Tidak Tamat SD': 2, 'Tidak Tamat SMP': 3, 'Tidak Tamat SMA': 4,
                          'Tidak Tamat Perguruan Tinggi': 5, 'Tamat Perguruan Tinggi': 6,
                          'Tamat Pasca Sarjana (S2/S3 - Master/Doktor)': 7}}
YES_NO_KEY = ['has_fridge', 'has_ac', 'has_water_heater', 'has_wifi', 'has_10g_gold', 'has_computer', 'has_motorcycle',
              'has_car', 'has_tv_32_inch_more', 'owns_land', 'electricity_capacity_1300_watt_or_more',
              'galloned_water', 'tattoo_history', 'has_cat', 'injection_drug_use', 'sex_worker']


def norm(s):
    return s.astype(str).str.split().str.join(' ').str.strip()


raw = pd.read_csv(RAW)
old = pd.read_csv(OLD)
raw.columns = [' '.join(str(c).split()) for c in raw.columns]
PRIVATE = ['Timestamp', raw.columns[1], 'Tanggal Lahir']           # timestamp, consent, date of birth
QCOLS = [c for c in raw.columns if c not in PRIVATE]                # the 70 questionnaire columns, in order
CODED = [c for c in old.columns if c != 'age']                      # the earlier copy's names, same order
assert len(QCOLS) == len(CODED) == 70, (len(QCOLS), len(CODED))
NAME = dict(zip(QCOLS, CODED))                                      # raw question text -> short name
assert raw[raw.columns[1]].nunique() == 1, 'everyone should have consented'
print(f'Loaded raw {raw.shape} and earlier copy {old.shape}; 70 questions line up by position.')

# ---------------------------------------------------------------- step 1: learn answer text -> code
def fingerprint(df, is_raw):
    k = (df['Titer'] if is_raw else df['titer']).astype(float).round(2).astype(str)
    for name in YES_NO_KEY:
        col = [q for q in QCOLS if NAME[q] == name][0] if is_raw else name
        v = norm(df[col]).isin(['Ya', 'a. Ya']).astype(int) if is_raw else df[col]   # both spellings of "yes"
        k = k + '|' + v.astype(str)
    return k


raw['_fp'], old['_fp'] = fingerprint(raw, True), fingerprint(old, False)
once = set(raw['_fp'].value_counts()[lambda s: s == 1].index) & set(old['_fp'].value_counts()[lambda s: s == 1].index)
pairs = old[old['_fp'].isin(once)].merge(raw[raw['_fp'].isin(once)], on='_fp')
CODEMAP = {}
for q in QCOLS:
    name = NAME[q]
    if name in TEXT or name == 'titer':
        continue
    t = pd.crosstab(norm(pairs[q]), pairs[name])
    assert (t.gt(0).sum(axis=1) == 1).all(), f'{name}: an answer text maps to more than one code'
    CODEMAP[name] = t.idxmax(axis=1).to_dict()
    CODEMAP[name].pop('nan', None)   # a blank answer stays missing (the earlier copy wrote 0 = "no soil contact" for
                                     # one person who reports soil contact 2-3 times a month)
    if name in EXPLICIT:
        clash = {k: v for k, v in CODEMAP[name].items() if EXPLICIT[name].get(k) != v}
        assert not clash, f'{name}: questionnaire codes disagree with the data: {clash}'
        CODEMAP[name] = dict(EXPLICIT[name])
    unseen = set(norm(raw[q])) - set(CODEMAP[name]) - {'nan'}
    if unseen:   # answers no safe pair gave: allowed only for plain numbers (counts), which map to themselves
        assert all(u.lstrip('-').isdigit() for u in unseen), f'{name}: unseen answers {sorted(unseen)[:3]}'
        CODEMAP[name].update({u: int(u) for u in unseen})
print(f'Step 1: {len(pairs)} safe pairs; {len(CODEMAP)} coded questions, every answer text has exactly one code.')

# ---------------------------------------------------------------- step 2: convert all rows and check against the copy
conv = pd.DataFrame({NAME[q]: norm(raw[q]).map(CODEMAP[NAME[q]]) for q in QCOLS if NAME[q] in CODEMAP})
missing = conv.isna().sum()
assert missing.sum() <= 1, missing[missing > 0]           # one blank answer in the raw file (handwash after soil)
conv['titer'] = raw['Titer'].astype(float)
for name in TEXT:
    conv[name] = norm(raw[[q for q in QCOLS if NAME[q] == name][0]])
cmp_cols = [c for c in CODED if c not in TEXT]
from collections import Counter
row_text = lambda df: df[cmp_cols].astype(float).round(2).fillna(-99).astype(str).agg('|'.join, axis=1)
a, b = row_text(conv).tolist(), row_text(old).tolist()
n_diff = sum((Counter(a) - Counter(b)).values())          # converted rows with no identical row in the earlier copy
blank_row = conv[cmp_cols].isna().any(axis=1).sum()
print(f'Step 2: {len(conv)} rows converted; rows identical to the earlier copy: {len(a) - n_diff}/{len(a)} '
      f'(differences expected only where the raw answer is blank: {blank_row} row)')
assert n_diff <= blank_row, 'converted rows do not match the earlier cleaned copy'

# ---------------------------------------------------------------- step 3: age at the interview date
born = pd.to_datetime(raw['Tanggal Lahir'], format='%m/%d/%Y', errors='coerce')
seen = pd.to_datetime(raw['Timestamp'], errors='coerce')
assert seen.notna().all() and born.notna().all(), 'a date did not parse'
age = np.floor((seen - born).dt.days / 365.25)
bad_age = ~age.between(15, 90)
conv['age'] = age.where(~bad_age)
old_age_by_fp = old.drop_duplicates('_fp', keep=False).set_index('_fp')['age']
shift = (old_age_by_fp.reindex(raw['_fp']).to_numpy() - conv['age'].to_numpy())
shift = pd.Series(shift).dropna()
print(f'Step 3: interviews {seen.min():%Y-%m} to {seen.max():%Y-%m}; impossible birth dates set to missing: '
      f'{int(bad_age.sum())}; age {conv["age"].min():.0f}-{conv["age"].max():.0f} (median {conv["age"].median():.0f}). '
      f'Earlier copy minus new age: {shift.value_counts().sort_index().to_dict()}')

# ---------------------------------------------------------------- step 4: restructure
clean = pd.DataFrame({'id': [f'J{i:03d}' for i in range(1, len(conv) + 1)]})
clean['igg_positive'] = (conv['serostatus'] == 1).astype(int)
clean['igg_borderline'] = (conv['serostatus'] == 2).astype(int)
clean['igg_titer'] = conv['titer']
feature_cols = [c for c in CODED if c not in ('titer', 'serostatus')]
for c in feature_cols + ['age']:
    if c not in clean:
        clean[c] = conv[c]
for c in FOLLOW_UP:
    clean[c] = clean[c].replace(-1, np.nan)
for c in MIXED_UP + MIXED_DOWN:
    code = clean[c]
    clean[c + '_applies'] = (code > 0).astype(int).where(code.notna())
    risk = code if c in MIXED_UP else 5 - code
    clean[c + '_risk'] = risk.where(code > 0)
assert clean['igg_positive'].sum() == 123 and clean['igg_borderline'].sum() == 1
assert not set(PRIVATE) & set(clean.columns)
assert ((clean['igg_titer'] <= 6.51) | (clean['igg_positive'] == 1) | (clean['igg_borderline'] == 1)).all()
print(f'Step 4: {clean.shape[1]} columns; 123 positive, 63 negative, 1 borderline; '
      f'-1 -> missing in {len(FOLLOW_UP)} follow-up questions; {len(MIXED_UP + MIXED_DOWN)} mixed scales split.')

# ---------------------------------------------------------------- step 5: save + data dictionary
os.makedirs(os.path.dirname(OUT), exist_ok=True)
clean.to_csv(OUT, index=False)
DESC = {
    'id': ('id', 'Row id J001-J187 in raw-file order (no link to any personal data)'),
    'igg_positive': ('target', 'Toxoplasma IgG positive (serostatus 1); borderline coded 0'),
    'igg_borderline': ('flag', 'Borderline IgG (1 person, titer 9.64); exclude as in the Makassar analysis'),
    'igg_titer': ('do not use', 'IgG titer from the same blood test (leaks the target)'),
    'age': ('feature', 'Age in whole years at the interview date (missing for 7 impossible birth dates)'),
    'gender': ('feature', '0 female, 1 male, 2 transgender woman (waria)'),
    'education': ('feature', '1 no school, 2 not finished elementary, 3 not finished junior high, 4 not finished senior '
                             'high, 5 not finished university, 6 university graduate, 7 postgraduate'),
    'job': ('text', 'Free-text occupation (not cleaned)'),
    'healthcare_access': ('text', 'Free-text ARV clinic name (not cleaned); possible grouping variable'),
    'drinking_water_source': ('feature', '0 branded bottled, 1 refill gallon, 2 tap (PAM), 3 well/river/ground'),
    'clean_water_source': ('feature', 'Water for bathing/washing (questionnaire code)'),
}
rows = []
for c in clean.columns:
    if c in DESC:
        role, desc = DESC[c]
    elif c.endswith('_applies'):
        role, desc = 'feature', f'1 if the question applies (original code > 0), else 0 ({c[:-8]})'
    elif c.endswith('_risk'):
        role, desc = 'feature', f'Risk among people it applies to: 1 safest ... 4 riskiest; missing if not applicable ({c[:-5]})'
    elif c in MIXED_UP:
        role, desc = 'original code', '0 not applicable, 1 always (safe) ... 4 never (risky); use the _applies/_risk columns'
    elif c in MIXED_DOWN:
        role, desc = 'original code', '0 does not eat raw, 4 always washes ... 1 never washes; use the _applies/_risk columns'
    elif c in FOLLOW_UP:
        role, desc = 'feature', 'Follow-up question; missing = skipped because it does not apply (was -1)'
    elif set(clean[c].dropna().unique()) <= {0, 1}:
        role, desc = 'feature', '1 yes, 0 no'
    else:
        role, desc = 'feature', 'Questionnaire code or count (see the questionnaire); counts not capped'
    rows.append({'column': c, 'role': role, 'description': desc,
                 'values': f'{clean[c].min()}-{clean[c].max()}' if clean[c].dtype != object else 'text',
                 'missing': int(clean[c].isna().sum())})
pd.DataFrame(rows).to_csv(DICT, index=False)
print(f'Step 5: saved {OUT} ({clean.shape[0]} rows x {clean.shape[1]} columns) and {DICT} ({len(rows)} rows).')

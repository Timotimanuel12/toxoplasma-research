"""
NHANES III (1988-1994) adult Toxoplasma dataset with DIRECT exposure questions
==============================================================================

Continuous NHANES (1999+) never asked about cats, gardening or meat-eating frequency.
NHANES III did. Toxoplasma IgG (TOP, IU/mL) was tested in phase 2 (1991-1994).
CDC lab documentation: TOP >= 7 IU/mL = positive ("titer below 7 IU/mL ... negative").

Inputs  (downloaded from CDC on first run into data/raw/NHANES3/):
  adult.dat + adult.sas   Household Adult Data File
  lab.dat   + lab.sas     Laboratory Data File
Output: data/processed/nhanes3_adult_toxo.csv
"""
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhanes3_io import read  # noqa: E402

RAW = os.path.join('data', 'raw', 'NHANES3')
OUT_CSV = os.path.join('data', 'processed', 'nhanes3_adult_toxo.csv')
URL = 'https://wwwn.cdc.gov/nchs/data/nhanes3/1a/{}'


def ensure_files():
    os.makedirs(RAW, exist_ok=True)
    for f in ['adult.dat', 'adult.sas', 'lab.dat', 'lab.sas']:
        p = os.path.join(RAW, f)
        if not os.path.exists(p):
            print(f'  downloading {f} from CDC ...')
            urllib.request.urlretrieve(URL.format(f), p)


def clean(s, bad):
    """Blank-but-applicable / don't-know codes -> NaN."""
    return s.where(~s.isin(bad))


def yn(s):
    return s.map({1: 1.0, 2: 0.0})


ADULT_COLS = ['SEQN', 'HSAGEIR', 'HSSEX', 'DMARETHN', 'DMPPIR', 'DMPCREGN', 'DMPMETRO', 'HSHSIZER',
              'HFLANG', 'HFA6XCR', 'HFA8R', 'HFE1', 'HFE2', 'HFE3', 'HFE4', 'HFE5', 'HFE7', 'HFE8A',
              'HFE8B', 'HFF4', 'HAR1', 'HAR3', 'HAT14', 'HAT15S', 'HAS17R', 'HAS19R',
              'HAN2CS', 'HAN2DS', 'HAN2ES', 'HAN2FS', 'HAN2GS', 'HAN2HS', 'HAN2IS', 'HAN4HS',
              'WTPFEX6', 'SDPPSU6', 'SDPSTRA6']
LAB_COLS = ['SEQN', 'TOP', 'AHP', 'HBP', 'HCP', 'H1P', 'H2P', 'HPP', 'EOP', 'PBP', 'WCP']


def build():
    ensure_files()
    a = read(os.path.join(RAW, 'adult.dat'), os.path.join(RAW, 'adult.sas'), ADULT_COLS)
    lab = read(os.path.join(RAW, 'lab.dat'), os.path.join(RAW, 'lab.sas'), LAB_COLS)
    df = lab.merge(a, on='SEQN', how='inner')
    df = df[(df['HSAGEIR'] >= 18) & df['TOP'].between(0, 240)].copy()   # 888 = not tested
    out = pd.DataFrame({'SEQN': df['SEQN']})

    # ---------------- target ----------------
    out['target_toxo_pos'] = (df['TOP'] >= 7).astype(int)

    # ---------------- baseline-equivalent features (same ideas as the 2013-14 model) ----------
    out['age'] = df['HSAGEIR']
    out['male'] = (df['HSSEX'] == 1).astype(float)
    out['race'] = df['DMARETHN']                          # 1 NH white, 2 NH black, 3 Mexican-Am, 4 other
    out['birthplace'] = clean(df['HFA6XCR'], [8, 9])      # 1 US, 2 Mexico, 3 other country
    out['education_years'] = clean(df['HFA8R'], [88, 99])
    out['poverty_income_ratio'] = clean(df['DMPPIR'], [888888])
    out['spanish_interview'] = df['HFLANG'].map({1: 0.0, 2: 1.0, 3: 0.0})
    smoke = pd.Series(np.nan, index=df.index)
    smoke[df['HAR1'] == 2] = 0
    smoke[(df['HAR1'] == 1) & (df['HAR3'] == 2)] = 1
    smoke[(df['HAR1'] == 1) & (df['HAR3'] == 1)] = 2
    out['smoking_status'] = smoke

    # ---------------- DIRECT exposure questions (not available after 1994) ----------------
    pets = clean(df['HFE7'], [8])
    out['cat_in_home'] = np.where(pets == 2, 0.0, yn(clean(df['HFE8B'], [8])))
    out['dog_in_home'] = np.where(pets == 2, 0.0, yn(clean(df['HFE8A'], [8])))
    garden = df['HAT15S'].where(df['HAT15S'] <= 100)            # 8888/9999/other codes -> NaN
    out['garden_yard_times_month'] = np.where(df['HAT14'] == 2, 0.0, garden)
    out['region'] = df['DMPCREGN']                         # 1 NE, 2 MW, 3 S, 4 W
    out['metro_1m_plus'] = (df['DMPMETRO'] == 1).astype(float)
    tap = clean(df['HFE5'], [8, 9])
    out['well_or_spring_water'] = np.where(df['HFE4'] == 1, 0.0, tap.map({1: 0.0, 2: 1.0, 3: 1.0}))
    out['bottled_water'] = clean(df['HFE4'], [8, 9]).map({1: 1.0, 2: 0.0})
    out['house_built'] = clean(df['HFE1'], [8, 9])         # 1 <1946, 2 1946-73, 3 1974+
    rooms = clean(df['HFE2'], [88, 99])
    out['household_size'] = df['HSHSIZER']
    out['crowding'] = df['HSHSIZER'] / rooms
    yrs = clean(df['HFE3'], [888, 999]).replace(777, 0.5)
    out['years_at_address'] = yrs
    out['food_insufficiency'] = clean(df['HFF4'], [8]) - 1  # 0 enough, 1 sometimes, 2 often not enough
    occ = clean(df['HAS17R'], [88, 99])
    ind = clean(df['HAS19R'], [88, 99])
    out['work_farm'] = (occ.isin([25, 26, 27]) | ind.isin([1, 2])).astype(float)
    out['work_soil_construction'] = (occ.isin([30, 37]) | ind.isin([3, 4])).astype(float)
    out['work_food_prep'] = (occ.isin([19, 20, 21]) | ind.isin([5])).astype(float)
    ffq = {'HAN2CS': ('processed_meat_per_month', [8888, 9999]),
           'HAN2DS': ('liver_organ_meat_per_month', [888, 999]),
           'HAN2ES': ('beef_per_month', [888, 999]),
           'HAN2FS': ('pork_per_month', [888, 999]),
           'HAN2GS': ('shellfish_per_month', [888, 999]),
           'HAN2HS': ('fish_per_month', [888, 999]),
           'HAN2IS': ('chicken_per_month', [888, 999]),
           'HAN4HS': ('salad_per_month', [888, 999])}
    for col, (name, bad) in ffq.items():
        out[name] = clean(df[col], bad)

    # ---------------- lab markers (other infections, blood) ----------------
    def lab_pn(s):
        return s.map({1: 1.0, 2: 0.0})
    out['hav_antibody'] = lab_pn(df['AHP'])
    out['hbv_core_antibody'] = lab_pn(df['HBP'])
    out['hcv_antibody'] = lab_pn(df['HCP'])
    out['hsv1'] = lab_pn(df['H1P'])
    out['hsv2'] = lab_pn(df['H2P'])
    out['h_pylori'] = lab_pn(df['HPP'])
    out['eosinophil_pct'] = clean(df['EOP'], [88])
    out['blood_lead'] = clean(df['PBP'], [8888])

    out['WTPFEX6'] = df['WTPFEX6']   # survey design (not features)
    out['SDPPSU6'] = df['SDPPSU6']
    out['SDPSTRA6'] = df['SDPSTRA6']
    return out.reset_index(drop=True)


BASE3 = ['age', 'male', 'race', 'birthplace', 'education_years', 'poverty_income_ratio',
         'spanish_interview', 'smoking_status']
EXPOSURE3 = ['cat_in_home', 'dog_in_home', 'garden_yard_times_month', 'region', 'metro_1m_plus',
             'well_or_spring_water', 'bottled_water', 'house_built', 'household_size', 'crowding',
             'years_at_address', 'food_insufficiency', 'work_farm', 'work_soil_construction',
             'work_food_prep', 'processed_meat_per_month', 'liver_organ_meat_per_month',
             'beef_per_month', 'pork_per_month', 'shellfish_per_month', 'fish_per_month',
             'chicken_per_month', 'salad_per_month']
LAB3 = ['hav_antibody', 'hbv_core_antibody', 'hcv_antibody', 'hsv1', 'hsv2', 'h_pylori',
        'eosinophil_pct', 'blood_lead']
CATEGORICAL3 = ['race', 'birthplace', 'region', 'house_built', 'smoking_status']


def main():
    df = build()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved {OUT_CSV}: {df.shape}, {df["target_toxo_pos"].mean():.1%} positive')
    return df


if __name__ == '__main__':
    main()

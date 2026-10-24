"""
Biological / exposure risk factors for Toxoplasma gondii from NHANES
=====================================================================

Two outputs:

1. data/processed/nhanes_adult_toxo_bio.csv
   The existing 2013-2014 adult cohort (nhanes_adult_toxo_clean.csv, n = 5,006, built by
   scripts/cleaning2.py) + 44 new risk-factor columns joined on SEQN.

2. data/processed/nhanes_adult_toxo_bio_2009_2014.csv
   The same cohort definition rebuilt for the three NHANES cycles that used the SAME
   Toxoplasma IgG assay and cutoff (Bio-Rad EIA, positive >= 33 IU/mL):
   2009-2010 (F), 2011-2012 (G), 2013-2014 (H). About 3x more participants.

All files are official CDC NHANES XPT files. Missing files are downloaded on first run into
data/raw/NHANES (2013-14), data/raw/NHANES_G (2011-12) and data/raw/NHANES_F (2009-10).
"""
import os
import urllib.request

import numpy as np
import pandas as pd

BASE_CSV = os.path.join('data', 'processed', 'nhanes_adult_toxo_clean.csv')
OUT_CSV = os.path.join('data', 'processed', 'nhanes_adult_toxo_bio.csv')
OUT_POOLED_CSV = os.path.join('data', 'processed', 'nhanes_adult_toxo_bio_2009_2014.csv')

CYCLES = {  # suffix: (folder, first year, label)
    'F': (os.path.join('data', 'raw', 'NHANES_F'), 2009, '2009-2010'),
    'G': (os.path.join('data', 'raw', 'NHANES_G'), 2011, '2011-2012'),
    'H': (os.path.join('data', 'raw', 'NHANES'), 2013, '2013-2014'),
}
CDC_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt'

# file stem -> which cycles have it (cotinine is COT in 2013-14 and COTNAL before)
FILES = ['SSTOXO', 'SSTOCA', 'DEMO', 'HSQ', 'SLQ', 'DPQ', 'SMQ', 'ALQ',
         'HEPA', 'HEPE', 'HEPBD', 'HEPC', 'HSV', 'HIV', 'CBC', 'PBCD', 'COT', 'COTNAL',
         'IMQ', 'DR1TOT', 'DR1IFF', 'HOQ', 'OCQ', 'FSQ', 'DUQ', 'HIQ', 'OHXDEN']
NOT_IN_CYCLE = {('SSTOCA', 'F'), ('COT', 'F'), ('COT', 'G'), ('COTNAL', 'H')}


# ----------------------------------------------------------------------------- I/O helpers
def ensure_files(suffixes=('F', 'G', 'H')):
    for s in suffixes:
        folder, year, _ = CYCLES[s]
        os.makedirs(folder, exist_ok=True)
        for stem in FILES:
            if (stem, s) in NOT_IN_CYCLE:
                continue
            name = f'{stem}_{s}'
            path = os.path.join(folder, f'{name}.xpt')
            if not os.path.exists(path):
                print(f'  downloading {name}.xpt from CDC ...')
                urllib.request.urlretrieve(CDC_URL.format(year=year, name=name), path)


def read(stem, s, cols):
    """Read the requested columns (those that exist in this cycle) and fix the XPT zero artifact."""
    if (stem, s) in NOT_IN_CYCLE:
        return pd.DataFrame(columns=['SEQN'] + cols)
    df = pd.read_sas(os.path.join(CYCLES[s][0], f'{stem}_{s}.xpt'), format='xport')
    df = df[['SEQN'] + [c for c in cols if c in df.columns]].copy()
    for c in cols:
        if c not in df:
            df[c] = np.nan
    for c in df.select_dtypes('float').columns:
        tiny = (df[c].abs() > 0) & (df[c].abs() < 1e-70)   # pandas reads a stored 0 as 5.4e-79
        df.loc[tiny, c] = 0.0
    return df


def pos_neg(s):
    """NHANES serology: 1 = positive, 2 = negative, 3 = indeterminate -> 1 / 0 / NaN."""
    return s.map({1.0: 1.0, 2.0: 0.0})


yes_no = pos_neg      # questionnaire: 1 = yes, 2 = no, 7/9 = refused/don't know -> 1 / 0 / NaN


# ------------------------------------------------------------- baseline (same as cleaning2.py)
def edu3_to_edu2(x):
    if pd.isna(x) or x in (77, 99):
        return np.nan
    if x <= 8 or x in (55, 66):
        return 1.0
    if x <= 12:
        return 2.0
    if x in (13, 14):
        return 3.0
    if x == 15:
        return 4.0
    return np.nan


def build_baseline(s):
    """Rebuild the cleaning2.py cohort and features for one cycle."""
    phq = [f'DPQ0{i}0' for i in range(1, 10)]
    toxo = read('SSTOXO', s, ['SSTOXG', 'SSTOXO'])
    toxo['igg'] = toxo['SSTOXG'].fillna(toxo['SSTOXO'])     # 2009-10 calls the column SSTOXO
    demo = read('DEMO', s, ['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'RIDRETH3', 'DMDBORN4', 'DMDBORN2',
                            'DMDYRSUS', 'DMDEDUC2', 'DMDEDUC3', 'INDFMPIR',
                            'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'])
    tables = [toxo[['SEQN', 'igg']], demo,
              read('HSQ', s, ['HSD010']), read('SLQ', s, ['SLD010H', 'SLQ050']),
              read('DPQ', s, phq), read('SMQ', s, ['SMQ020', 'SMQ040']),
              read('ALQ', s, ['ALQ101', 'ALQ110', 'ALQ120Q', 'ALQ130'])]
    df = tables[0]
    for t in tables[1:]:
        df = df.merge(t, on='SEQN', how='inner')
    df = df[(df['RIDAGEYR'] >= 18) & df['igg'].notna()].copy()

    codes = {'DMDBORN4': [77, 99], 'DMDBORN2': [7, 9], 'DMDYRSUS': [77, 99], 'DMDEDUC2': [7, 9],
             'DMDEDUC3': [77, 99], 'HSD010': [7, 9], 'SLD010H': [77, 99], 'SLQ050': [7, 9],
             'SMQ020': [7, 9], 'SMQ040': [7, 9], 'ALQ101': [7, 9], 'ALQ110': [7, 9],
             'ALQ120Q': [777, 999], 'ALQ130': [77, 99, 777, 999]}
    codes.update({p: [7, 9] for p in phq})
    for col, bad in codes.items():
        df.loc[df[col].isin(bad), col] = np.nan

    df['smoking_status'] = np.nan
    df.loc[df['SMQ020'] == 2, 'smoking_status'] = 0
    df.loc[(df['SMQ020'] == 1) & (df['SMQ040'] == 3), 'smoking_status'] = 1
    df.loc[(df['SMQ020'] == 1) & df['SMQ040'].isin([1, 2]), 'smoking_status'] = 2

    df['drinks_per_day'] = df['ALQ130']
    non_drinker = ((df['ALQ101'] == 2) & (df['ALQ110'] == 2)) | (df['ALQ120Q'] == 0)
    df.loc[non_drinker, 'drinks_per_day'] = 0.0

    # born in the US: DMDBORN4 (2011+) or DMDBORN2 (2009-10), code 1 = 50 US states / DC
    born = df['DMDBORN4'].fillna(df['DMDBORN2'])
    df['DMDBORN4'] = np.where(born == 1, 1.0, np.where(born.notna(), 2.0, np.nan))
    df['yrs_in_us_grp'] = np.nan
    df.loc[df['DMDBORN4'] == 1, 'yrs_in_us_grp'] = 0
    df.loc[df['DMDYRSUS'].isin([1, 2, 3]), 'yrs_in_us_grp'] = 1
    df.loc[df['DMDYRSUS'].isin([4, 5]), 'yrs_in_us_grp'] = 2
    df.loc[df['DMDYRSUS'].isin([6, 7, 8, 9]), 'yrs_in_us_grp'] = 3
    df.loc[(df['DMDBORN4'] == 2) & df['DMDYRSUS'].isna(), 'yrs_in_us_grp'] = 4

    young = df['DMDEDUC2'].isna() & df['RIDAGEYR'].isin([18, 19])
    df.loc[young, 'DMDEDUC2'] = df.loc[young, 'DMDEDUC3'].map(edu3_to_edu2)
    df['phq9_total_score'] = df[phq].sum(axis=1, skipna=False)

    toca = read('SSTOCA', s, ['SSTCAR'])
    df = df.merge(toca, on='SEQN', how='left')
    df['toxocara_pos'] = df['SSTCAR'].map({1.0: 1, 2.0: 0})

    df['target_toxo_pos'] = (df['igg'] >= 33).astype(int)
    keep = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'RIDRETH3', 'DMDBORN4', 'DMDEDUC2',
            'INDFMPIR', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA', 'HSD010', 'SLD010H', 'SLQ050',
            'smoking_status', 'drinks_per_day', 'yrs_in_us_grp', 'toxocara_pos',
            'phq9_total_score', 'target_toxo_pos']
    return df[keep]


# --------------------------------------------------------------------- new risk factors
def build_new_features(s):
    f = {}

    # ===== LAB: other infections that travel the same routes as Toxoplasma =====
    t = read('HEPA', s, ['LBXHA']).merge(read('IMQ', s, ['IMQ011']), on='SEQN', how='outer')
    t['hav_antibody'] = pos_neg(t['LBXHA'])
    vaccinated = t['IMQ011'].isin([1.0, 2.0])
    t['hav_vaccinated'] = np.where(vaccinated, 1.0, np.where(t['IMQ011'] == 3.0, 0.0, np.nan))
    # antibodies can come from the vaccine: "natural infection" = antibody + and not vaccinated
    t['hav_natural_infection'] = np.where(t['hav_antibody'].isna(), np.nan,
                                          ((t['hav_antibody'] == 1) & ~vaccinated).astype(float))
    f['hav'] = t[['SEQN', 'hav_antibody', 'hav_vaccinated', 'hav_natural_infection']]

    t = read('HEPE', s, ['LBDHEG'])
    t['hev_igg'] = pos_neg(t['LBDHEG'])
    f['hev'] = t[['SEQN', 'hev_igg']]

    t = read('HEPBD', s, ['LBXHBC'])
    t['hbv_core_antibody'] = pos_neg(t['LBXHBC'])
    f['hbv'] = t[['SEQN', 'hbv_core_antibody']]

    t = read('HEPC', s, ['LBXHCR', 'LBDHCV'])
    if s == 'H':   # 2013-14: 1/2 = antibody + (RNA +/-), 3 = antibody negative
        t['hcv_antibody'] = t['LBXHCR'].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})
    else:          # 2009-12: LBDHCV 1 = positive, 2 = negative, 5 = screen + / not confirmed
        t['hcv_antibody'] = t['LBDHCV'].map({1.0: 1.0, 2.0: 0.0, 5.0: 0.0})
    f['hcv'] = t[['SEQN', 'hcv_antibody']]

    t = read('HSV', s, ['LBXHE1', 'LBXHE2'])          # tested at ages 14-49 only
    t['hsv1'] = pos_neg(t['LBXHE1'])
    t['hsv2'] = pos_neg(t['LBXHE2'])
    f['hsv'] = t[['SEQN', 'hsv1', 'hsv2']]

    t = read('HIV', s, ['LBDHI'])                     # tested at ages 18-59 only
    t['hiv'] = pos_neg(t['LBDHI'])
    f['hiv'] = t[['SEQN', 'hiv']]

    t = read('SSTOCA', s, ['SSTCAMFI'])               # Toxocara antibody level (2011-14 only)
    t['toxocara_mfi_log'] = np.log1p(t['SSTCAMFI'].astype(float).clip(lower=0))
    f['toca'] = t[['SEQN', 'toxocara_mfi_log']]

    # ===== LAB: immune / blood markers =====
    t = read('CBC', s, ['LBXWBCSI', 'LBXEOPCT', 'LBXLYPCT', 'LBXMOPCT', 'LBXHGB'])
    f['cbc'] = t.rename(columns={'LBXWBCSI': 'wbc_count', 'LBXEOPCT': 'eosinophil_pct',
                                 'LBXLYPCT': 'lymphocyte_pct', 'LBXMOPCT': 'monocyte_pct',
                                 'LBXHGB': 'hemoglobin'})

    # ===== LAB: environmental exposure markers =====
    t = read('PBCD', s, ['LBXBPB', 'LBXBCD', 'LBXTHG'])
    t['blood_lead_log'] = np.log(t['LBXBPB'])          # soil, dust, older housing
    t['blood_cadmium_log'] = np.log(t['LBXBCD'])
    t['blood_mercury_log'] = np.log(t['LBXTHG'])       # fish / seafood intake
    f['metals'] = t[['SEQN', 'blood_lead_log', 'blood_cadmium_log', 'blood_mercury_log']]

    t = read('COT' if s == 'H' else 'COTNAL', s, ['LBXCOT'])
    t['cotinine_log'] = np.log10(t['LBXCOT'].astype(float).clip(lower=0.011))
    f['cot'] = t[['SEQN', 'cotinine_log']]

    # ===== QUESTIONNAIRE: water and food =====
    tot = read('DR1TOT', s, ['DR1TWS', 'DRD340', 'DRD350F', 'DRD360', 'DR1DRSTZ'])
    # DR1TWS: 1 community supply, 2 well/rain cistern, 3 spring, 4 don't drink tap water
    tot['well_or_spring_water'] = tot['DR1TWS'].map({1.0: 0.0, 2.0: 1.0, 3.0: 1.0, 4.0: 0.0})
    tot['no_tap_water'] = tot['DR1TWS'].map({1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 1.0})
    tot['shellfish_30d'] = yes_no(tot['DRD340'])
    tot['oysters_30d'] = yes_no(tot['DRD350F'])        # only asked if shellfish = yes
    tot.loc[tot['shellfish_30d'] == 0, 'oysters_30d'] = 0.0
    tot['fish_30d'] = yes_no(tot['DRD360'])
    f['tot'] = tot[['SEQN', 'well_or_spring_water', 'no_tap_water', 'shellfish_30d',
                    'oysters_30d', 'fish_30d']]

    # 24-hour recall: grams eaten yesterday by meat type (USDA food code, first two digits)
    iff = read('DR1IFF', s, ['DR1IFDCD', 'DR1IGRMS', 'DR1DRSTZ'])
    iff = iff[iff['DR1DRSTZ'] == 1]                    # reliable recalls only
    meat_groups = {21: 'beef_g', 22: 'pork_g', 23: 'lamb_veal_game_g', 24: 'poultry_g',
                   25: 'organ_sausage_lunchmeat_g', 26: 'fish_shellfish_g'}
    iff = iff.assign(grp=(iff['DR1IFDCD'] // 1_000_000).astype(int).map(meat_groups))
    meat = (iff.dropna(subset=['grp'])
               .pivot_table(index='SEQN', columns='grp', values='DR1IGRMS', aggfunc='sum'))
    reliable = tot.loc[tot['DR1DRSTZ'] == 1, ['SEQN']].set_index('SEQN')
    meat = reliable.join(meat).reindex(columns=list(meat_groups.values())).fillna(0.0)
    f['meat'] = np.log1p(meat).add_suffix('_log').reset_index()

    # ===== QUESTIONNAIRE: household, migration, work, poverty =====
    h = read('DEMO', s, ['DMDHHSIZ', 'SIALANG', 'DMDCITZN', 'DMQMILIZ', 'DMQMILIT', 'DMQADFC'])
    h = h.merge(read('HOQ', s, ['HOD050', 'HOQ065']), on='SEQN', how='left')
    h['household_size'] = h['DMDHHSIZ']
    h['crowding'] = h['household_size'] / h['HOD050'].where(h['HOD050'] <= 13)   # people per room
    h['home_rented'] = h['HOQ065'].map({1.0: 0.0, 2.0: 1.0, 3.0: 1.0})
    h['spanish_interview'] = h['SIALANG'].map({1.0: 0.0, 2.0: 1.0})
    h['not_us_citizen'] = h['DMDCITZN'].map({1.0: 0.0, 2.0: 1.0})
    military = h['DMQMILIZ'].fillna(h['DMQMILIT'])
    h['military_abroad'] = np.where(military == 2, 0.0, h['DMQADFC'].map({1.0: 1.0, 2.0: 0.0}))
    f['house'] = h[['SEQN', 'household_size', 'crowding', 'home_rented', 'spanish_interview',
                    'not_us_citizen', 'military_abroad']]

    t = read('OCQ', s, ['OCD231', 'OCD391', 'OCD241', 'OCD392'])
    ind = t['OCD231'].fillna(t['OCD391'])              # current job, otherwise longest job
    occ = t['OCD241'].fillna(t['OCD392'])
    t['work_agriculture'] = ((ind == 1) | (occ == 18)).astype(float)        # farming/fishing/forestry
    t['work_food_service'] = ((ind == 18) | (occ == 13)).astype(float)      # food prep / serving
    t['work_construction_soil'] = ((ind == 4) | occ.isin([14, 19])).astype(float)  # soil contact
    f['ocq'] = t[['SEQN', 'work_agriculture', 'work_food_service', 'work_construction_soil']]

    t = read('FSQ', s, ['FSDAD'])
    t['food_insecurity'] = t['FSDAD'] - 1              # 0 = full ... 3 = very low food security
    f['fsq'] = t[['SEQN', 'food_insecurity']]

    t = read('HIQ', s, ['HIQ011'])
    t['no_health_insurance'] = t['HIQ011'].map({1.0: 0.0, 2.0: 1.0})
    f['hiq'] = t[['SEQN', 'no_health_insurance']]

    t = read('DUQ', s, ['DUQ370', 'DUQ240'])          # asked at ages 18-69 only
    t['inject_drugs_ever'] = yes_no(t['DUQ370'])
    t['hard_drugs_ever'] = yes_no(t['DUQ240'])
    t.loc[t['hard_drugs_ever'] == 0, 'inject_drugs_ever'] = 0.0
    f['duq'] = t[['SEQN', 'inject_drugs_ever', 'hard_drugs_ever']]

    # ===== EXAM: oral health -> number of missing permanent teeth (28, no wisdom teeth) =====
    teeth = [f'OHX{i:02d}TC' for i in range(2, 32) if i not in (16, 17)]
    t = read('OHXDEN', s, ['OHDDESTS'] + teeth)
    t['missing_teeth'] = np.where(t['OHDDESTS'] == 1, (t[teeth] == 4).sum(axis=1), np.nan)
    f['oh'] = t[['SEQN', 'missing_teeth']]

    out = None
    for df in f.values():
        df = df.astype({'SEQN': float})
        out = df if out is None else out.merge(df, on='SEQN', how='outer')
    return out


# ------------------------------------------------------------------------ feature groups
LAB_INFECTION = ['hav_natural_infection', 'hav_antibody', 'hev_igg', 'hbv_core_antibody',
                 'hcv_antibody', 'hsv1', 'hsv2', 'hiv', 'toxocara_mfi_log']
LAB_BLOOD = ['wbc_count', 'eosinophil_pct', 'lymphocyte_pct', 'monocyte_pct', 'hemoglobin',
             'blood_lead_log', 'blood_cadmium_log', 'blood_mercury_log', 'cotinine_log']
Q_FOOD_WATER = ['well_or_spring_water', 'no_tap_water', 'shellfish_30d', 'oysters_30d',
                'fish_30d', 'beef_g_log', 'pork_g_log', 'lamb_veal_game_g_log', 'poultry_g_log',
                'organ_sausage_lunchmeat_g_log', 'fish_shellfish_g_log']
Q_LIVING = ['household_size', 'crowding', 'home_rented', 'spanish_interview', 'not_us_citizen',
            'military_abroad', 'work_agriculture', 'work_food_service', 'work_construction_soil',
            'food_insecurity', 'no_health_insurance', 'inject_drugs_ever', 'hard_drugs_ever',
            'hav_vaccinated', 'missing_teeth']
NEW_FEATURES = LAB_INFECTION + LAB_BLOOD + Q_FOOD_WATER + Q_LIVING


def build_2013_2014():
    ensure_files(('H',))
    base = pd.read_csv(BASE_CSV)
    df = base.merge(build_new_features('H'), on='SEQN', how='left')
    assert len(df) == len(base), 'row count changed during merge'
    return df


def build_pooled():
    ensure_files(('F', 'G', 'H'))
    parts = []
    for s, (_, _, label) in CYCLES.items():
        b = build_baseline(s).astype({'SEQN': float})
        d = b.merge(build_new_features(s), on='SEQN', how='left')
        assert len(d) == len(b)
        d.insert(1, 'cycle', label)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    df = build_2013_2014()
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved {OUT_CSV}: {df.shape}')
    pooled = build_pooled()
    pooled.to_csv(OUT_POOLED_CSV, index=False)
    print(f'Saved {OUT_POOLED_CSV}: {pooled.shape}')
    print(pooled.groupby('cycle')['target_toxo_pos'].agg(['size', 'mean']))


if __name__ == '__main__':
    main()

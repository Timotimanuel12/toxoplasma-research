"""
Makassar pregnant-women Toxoplasma dataset -> clean CSV for modelling

Source: Polanunu, Wahyuni & Hamid (2021), "Seroprevalence and associated risk factors of
Toxoplasma gondii infection among pregnant mother in Makassar, Indonesia", PLoS ONE 16(6):
e0245572. Supporting file S1 (SPSS), CC BY 4.0 -> cite the paper when using the data.

Read SPSS file -> readable column names -> yes/no to 1/0 -> site from ID prefix ->
drop redundant columns -> target (IgG positive; borderline = negative, as in NHANES) ->
checks -> save CSV + data dictionary.

The file was checked against the paper with scripts/verify_against_paper.py: all IgG/IgM results,
education, site sizes and 14 of 15 rows of Table 3 match exactly (details in that script's output).
The paper excluded the 2 borderline IgG results from its analysis; use igg_borderline to do the same.

Run from inside makassar_project/.
"""
import os

import pandas as pd
import pyreadstat

RAW = os.path.join('data', 'raw', 'polanunu2021_S1.sav')
OUT_CSV = os.path.join('data', 'processed', 'makassar_clean.csv')
OUT_DICT = os.path.join('data', 'processed', 'makassar_data_dictionary.csv')

raw, meta = pyreadstat.read_sav(RAW)
print(f"Raw file: {raw.shape[0]} women, {raw.shape[1]} columns")

# ---- 1. Basic checks on the raw file -----------------------------------------
assert raw.isna().sum().sum() == 0, 'unexpected missing values'
assert not raw['Code'].duplicated().any(), 'duplicate IDs'
assert set(raw['ToxoIgG_Positivity'].unique()) <= {1, 2, 3}

# ---- 2. Build the clean table ------------------------------------------------
df = pd.DataFrame()
df['id'] = raw['Code']
# The ID prefix marks the primary health centre (Puskesmas); names and sizes match Table 2 of the paper
SITE_NAMES = {'SD': 'Sudiang', 'SR': 'Sudiang Raya', 'PC': 'Paccerakkang', 'KB': 'Kaluku Bodoa', 'AT': 'Antang',
              'KK': 'Kassi-Kassi', 'MG': 'Mangasa', 'JG': 'Jongaya', 'TT': 'Tamalate'}
df['site'] = raw['Code'].str.extract(r'^([A-Z]+)')[0]
df['site_name'] = df['site'].map(SITE_NAMES)

df['age'] = raw['age'].astype(int)
df['trimester'] = raw['Trimester'].astype(int)               # 1, 2, 3
df['gravidity'] = raw['Gravidity'].astype(int)               # number of pregnancies, incl. this one
df['parity'] = raw['Paritas'].astype(int)                    # number of previous births
df['education'] = raw['Education'].astype(int)               # 1 elementary ... 4 above senior high


def yes_no(col):
    """SPSS coding 1 = Yes, 2 = No  ->  1 / 0."""
    return raw[col].map({1.0: 1, 2.0: 0}).astype(int)


df['abortion_history'] = yes_no('AbortusHistory')
df['employed'] = yes_no('Employed')

# Water source. The "chlorinated / treated" columns in the raw file are fully determined by
# the source (pipe = chlorinated/treated, pump and surface = not), so only the source is kept.
df['washing_water'] = raw['WFW'].map({1.0: 'surface', 2.0: 'pump', 3.0: 'pipe'})
# One woman used "pipe and gallon" water for cooking; merged into "pipe" (both are treated)
df['cooking_water'] = raw['WFCP'].map({1.0: 'surface', 2.0: 'pump', 3.0: 'pipe', 4.0: 'gallon', 5.0: 'pipe'})
df['drinks_gallon_water'] = raw['WaterForDrinking'].map({1.0: 1, 2.0: 0}).astype(int)

# Exposure questions
df['raises_cat'] = yes_no('RaisingACat')
df['cat_contact'] = yes_no('ContactWithCat')
df['cleans_litter'] = yes_no('CleaningLittebox')
df['soil_contact'] = yes_no('ContactSoil2')
df['eats_chicken_satay'] = yes_no('ConsumeChickenSatay')
df['eats_raw_vegetables'] = yes_no('ConsumeRawVegetable')
df['drinks_unboiled_water'] = yes_no('ConsumeUnboiledWater')

# Target: IgG positive (1). The 2 borderline results are coded negative, like NHANES equivocal.
df['igg_positive'] = (raw['ToxoIgG_Positivity'] == 1).astype(int)
df['igg_borderline'] = (raw['ToxoIgG_Positivity'] == 3).astype(int)   # 2 women; the paper excluded them
# Antibody index from the same blood test: kept for reference only, NEVER a model feature
df['igg_index'] = raw['ToxoIgG_Index'].round(4)
# IgM is dropped: nobody was IgM positive (178 negative, 6 borderline) and it is also the test itself.

# ---- 3. Consistency checks ---------------------------------------------------
assert (df['parity'] <= df['gravidity'] - 1).all(), 'parity larger than previous pregnancies'
assert df.notna().all().all(), 'a recode produced missing values'
assert df.loc[df['igg_positive'] == 1, 'igg_index'].min() > 1.0, 'a positive result has an IgG index below 1.0'
# Answers that look odd but are kept as reported:
odd_litter = int(((df['cleans_litter'] == 1) & (df['cat_contact'] == 0)).sum())
odd_cat = int(((df['raises_cat'] == 1) & (df['cat_contact'] == 0)).sum())
print(f"Kept as reported: {odd_litter} clean cat litter but report no cat contact; "
      f"{odd_cat} raise a cat but report no cat contact")

# ---- 4. Summary --------------------------------------------------------------
n_pos = int(df['igg_positive'].sum())
print(f"\nClean table: {df.shape[0]} women, {df.shape[1]} columns")
print(f"IgG positive: {n_pos} ({n_pos / len(df):.1%}); "
      f"borderline coded negative: {int((raw['ToxoIgG_Positivity'] == 3).sum())}")
print("\nWomen and positivity per site:")
print(df.groupby('site')['igg_positive'].agg(n='size', positive='sum', rate='mean')
      .sort_values('n', ascending=False).round(2).to_string())
print("\nYes-rates of the exposure questions:")
expo = ['cat_contact', 'raises_cat', 'cleans_litter', 'soil_contact', 'eats_chicken_satay',
        'eats_raw_vegetables', 'drinks_unboiled_water', 'drinks_gallon_water']
print(df[expo].mean().round(2).to_string())

# ---- 5. Data dictionary and save --------------------------------------------
dictionary = pd.DataFrame([
    ('id', 'id', 'Participant code from the original file', 'text'),
    ('site', 'grouping', 'Primary health centre (ID prefix); use for grouped / leave-one-site-out CV', 'AT, JG, KB, KK, MG, PC, SD, SR, TT'),
    ('site_name', 'grouping', 'Name of the primary health centre (Puskesmas), as in Table 2 of the paper', '9 names'),
    ('age', 'feature', 'Age (years)', '15-42'),
    ('trimester', 'feature', 'Trimester of pregnancy', '1, 2, 3'),
    ('gravidity', 'feature', 'Number of pregnancies including the current one', '1-8'),
    ('parity', 'feature', 'Number of previous births', '0-6'),
    ('education', 'feature', 'Education level (ordinal)', '1 elementary, 2 junior high, 3 senior high, 4 above senior high'),
    ('abortion_history', 'feature', 'History of miscarriage/abortion', '1 yes, 0 no'),
    ('employed', 'feature', 'Employed', '1 yes, 0 no'),
    ('washing_water', 'feature', 'Water source for washing (pipe = chlorinated)', 'surface, pump, pipe'),
    ('cooking_water', 'feature', 'Water source for cooking (pipe/gallon = treated); 1 "pipe and gallon" merged into pipe', 'surface, pump, pipe, gallon'),
    ('drinks_gallon_water', 'feature', 'Drinking water: filtered refill-gallon water (paper: "filtered") vs boiled water', '1 filtered gallon, 0 boiled'),
    ('raises_cat', 'feature', 'Raises a cat', '1 yes, 0 no'),
    ('cat_contact', 'feature', 'Contact with cats', '1 yes, 0 no'),
    ('cleans_litter', 'feature', 'Cleans cat litter', '1 yes, 0 no'),
    ('soil_contact', 'feature', 'Contact with soil', '1 yes, 0 no'),
    ('eats_chicken_satay', 'feature', 'Ate undercooked meat (chicken satay) during pregnancy', '1 yes, 0 no'),
    ('eats_raw_vegetables', 'feature', 'Ate raw vegetables during pregnancy', '1 yes, 0 no'),
    ('drinks_unboiled_water', 'feature', 'Drank un-boiled / filtered water during pregnancy', '1 yes, 0 no'),
    ('igg_positive', 'target', 'Toxoplasma IgG positive, index > 1.1 (borderline coded negative)', '1 positive, 0 negative'),
    ('igg_borderline', 'flag', 'IgG borderline (index 0.9-1.1); the paper excluded these 2 women', '1 borderline, 0 not'),
    ('igg_index', 'do not use', 'IgG antibody index from the same blood test (leaks the target)', '0.09-3.01'),
], columns=['column', 'role', 'description', 'values'])
assert list(dictionary['column']) == list(df.columns)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
dictionary.to_csv(OUT_DICT, index=False)
print(f"\nSaved: {OUT_CSV}\nSaved: {OUT_DICT}")

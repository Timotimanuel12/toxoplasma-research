"""
Public individual-level Toxoplasma datasets (non-NHANES) with exposure questionnaires.

Sources: open-access papers whose supplementary files contain participant-level data
(found through Europe PMC; all CC BY). Each loader returns (X, y, info):
  X    feature DataFrame (questionnaire / demographic variables only - no Toxoplasma test results)
  y    1 = Toxoplasma IgG seropositive
  info dict with citation details
LEAKAGE RULE: every column measuring Toxoplasma itself (IgM, ELISA index, rapid test, avidity,
"currently / ever infected with toxo") is dropped.

Raw supplementary files are expected in data/external/<key>/ (see SOURCES for URLs).
"""
import os
import io
import zipfile

import numpy as np
import pandas as pd

EXT = os.path.join('data', 'external')

SOURCES = {
    'indonesia_pregnant': dict(pmc='PMC8177418', doi='10.1371/journal.pone.0245572',
                               cite='Polanunu, Wahyuni & Hamid 2021, PLoS One', setting='Pregnant women, Makassar, Indonesia',
                               file='pone.0245572.s001.sav'),
    'cameroon_hiv': dict(pmc='PMC8638881', doi='10.1371/journal.pone.0256947',
                         cite='PLoS One 2021', setting='People living with HIV, Bamenda, Cameroon',
                         file='pone.0256947.s001.xlsx'),
    'china_women': dict(pmc='PMC9760017', doi='10.7717/peerj.14569',
                        cite='PeerJ 2022', setting='Women with gynaecological tumour + matched controls, eastern China',
                        file='peerj-10-14569-s001.xlsx'),
    'myanmar_slaughter': dict(pmc='PMC10101469', doi='10.1371/journal.pone.0284352',
                              cite='PLoS One 2023', setting='Slaughterhouse workers, Yangon, Myanmar',
                              file='pone.0284352.s003.xlsx'),
    'iran_suicide': dict(pmc='PMC12210893', doi='10.1186/s12879-025-11200-1',
                         cite='BMC Infect Dis 2025', setting='Suicide attempters + controls, Lorestan, Iran',
                         file='12879_2025_11200_MOESM1_ESM.sav'),
    'philippines_cebu': dict(pmc='PMC6561560', doi='10.1371/journal.pone.0217989',
                             cite='PLoS One 2019', setting='General population, Cebu, Philippines',
                             file='pone.0217989.s006.xlsx'),
    'thai_border_pregnant': dict(pmc='PMC5508912', doi='10.4269/ajtmh.16-0999',
                                 cite='Am J Trop Med Hyg 2017', setting='Migrant/refugee pregnant women, Thailand-Myanmar border',
                                 file='tpmd160999.SD1.xlsx'),
    'brazil_homeless': dict(pmc='PMC7674916', doi='10.3389/fpubh.2020.580637',
                            cite='Front Public Health 2020', setting='Homeless persons, Sao Paulo, Brazil',
                            file='Table_1.xlsx'),
    'ghana_sickle_cell': dict(pmc='PMC13048495', doi='10.1371/journal.pone.0346155',
                              cite='PLoS One 2026', setting='Sickle cell disease patients, Ho, Ghana',
                              file='pone.0346155.s002.xlsx'),
}


def fetch(key):
    """Download the paper's supplementary zip from Europe PMC and extract the data file."""
    import requests
    s = SOURCES[key]
    folder = os.path.join(EXT, key)
    path = os.path.join(folder, s['file'])
    if os.path.exists(path):
        return path
    os.makedirs(folder, exist_ok=True)
    r = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{s['pmc']}/supplementaryFiles", timeout=300)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for n in z.namelist():
        if os.path.basename(n) == s['file']:
            open(path, 'wb').write(z.read(n))
            return path
        if n.lower().endswith('.zip'):
            inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
            for m in inner.namelist():
                if os.path.basename(m) == s['file']:
                    open(path, 'wb').write(inner.read(m))
                    return path
    raise FileNotFoundError(f"{s['file']} not in supplementary files of {s['pmc']}")


def yes(series, yes_values=('yes',)):
    s = series.astype(str).str.strip().str.lower()
    return np.where(series.isna(), np.nan, s.isin(yes_values).astype(float))


def code12(series):
    """SPSS-style 1 = yes, 2 = no -> 1 / 0."""
    return series.map({1: 1.0, 2: 0.0, 1.0: 1.0, 2.0: 0.0})


# ------------------------------------------------------------------ loaders
def indonesia_pregnant():
    import pyreadstat
    d, _ = pyreadstat.read_sav(fetch('indonesia_pregnant'))
    d = d[d['ToxoIgG_Positivity'].isin([1, 2])]                    # 3 = equivocal (2 women) dropped
    y = (d['ToxoIgG_Positivity'] == 1).astype(int)                  # 1 = positive (60 / 182 = 33%, as in paper)
    X = pd.DataFrame({'age': d['age'], 'trimester': d['Trimester'], 'parity': d['Paritas'],
                      'gravidity': d['Gravidity'], 'abortion_history': code12(d['AbortusHistory']),
                      'education': d['Education'], 'employed': code12(d['Employed']),
                      'water_washing_src': d['WFW'], 'water_cooking_src': d['WFCP'],
                      'water_drinking': d['WaterForDrinking'],
                      'cat_owner': code12(d['RaisingACat']), 'cat_contact': code12(d['ContactWithCat']),
                      'cleans_litterbox': code12(d['CleaningLittebox']), 'soil_contact': code12(d['ContactSoil2']),
                      'eats_chicken_satay': code12(d['ConsumeChickenSatay']),
                      'raw_vegetables': code12(d['ConsumeRawVegetable']),
                      'unboiled_water': code12(d['ConsumeUnboiledWater'])})
    return X.reset_index(drop=True), y.reset_index(drop=True)


def cameroon_hiv():
    d = pd.read_excel(fetch('cameroon_hiv'))
    d.columns = [str(c).strip() for c in d.columns]
    y = (d['IgG resutls'].str.strip().str.lower() == 'positive').astype(int)
    art = d['Duration on ART'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    prof = d['Profession'].astype(str).str.strip().str.lower()
    X = pd.DataFrame({'age': d['Age'], 'male': yes(d['Sex'], ('m',)),
                      'married': yes(d['Marital Status'], ('married',)),
                      'education': d['Level of Education'].str.strip().map({'no education': 0, 'primary': 1, 'sec': 2, 'university': 3}),
                      'farmer': (prof == 'farming').astype(float),
                      'years_on_art': art, 'art_defaulted': yes(d['Defaulted?']),
                      'raw_meat': yes(d['Eat raw meat?']), 'raw_meat_dont_know': yes(d['Eat raw meat?'], ("don't know",)),
                      'raw_vegetables': yes(d['Eat raw vegitable?']), 'cat_owner': yes(d['Own a cat']),
                      'neighbour_cat': yes(d['Does neighbour own a cat?']), 'garden': yes(d['Have a garden']),
                      'soil_contact': yes(d['Contact with garden soil']),
                      'pipe_water': d['Drinking water'].astype(str).str.contains('pipe').astype(float),
                      'spring_water': d['Drinking water'].astype(str).str.contains('spring').astype(float),
                      'open_water_container': d['Water storage'].astype(str).str.contains('open').astype(float),
                      'water_untreated': yes(d['Water Treatment'], ('not treated',)),
                      'symptoms': yes(d['Signs/Symptom']), 'cd4_count': d['CD4 T Cell Count']})
    return X, y


def china_women():
    p = fetch('china_women')
    t = pd.read_excel(p, sheet_name='patients with tumor')
    c = pd.read_excel(p, sheet_name='data for control')
    rows = []
    for d, tumour in [(t, 1.0), (c, 0.0)]:
        d.columns = [str(x).strip() for x in d.columns]
        rows.append(pd.DataFrame({
            'age': d['Age'], 'tumour_patient': tumour, 'rural': yes(d['Residence area'], ('rural',)),
            'cat_contact': yes(d['Contact with cats']), 'dog_contact': yes(d['Contact with dogs']),
            'swine_contact': yes(d['Contact with swine']), 'raw_meat': yes(d['Consumption of raw/undercooked meat']),
            'raw_vegetables': yes(d['Consumption of raw vegetables']), 'soil_contact': yes(d['Exposure to soil']),
            'drinking_water_flag': yes(d['Source of drinking water']), 'abortion_history': yes(d['History of abortion']),
            'farmer': yes(d['occupation'], ('farmer',)), 'y': yes(d['IgG'])}))
    d = pd.concat(rows, ignore_index=True)
    return d.drop(columns='y'), d['y'].astype(int)


def myanmar_slaughter():
    d = pd.read_excel(fetch('myanmar_slaughter'))
    y = (d['Seropositive'] == 1).astype(int)
    drop = ['ID', 'Test_Ig', 'Seropositive', 'Edu_2gp', 'O_Service_duration_cat', 'O_Job_2cat', 'Knowledge_level']
    X = d.drop(columns=drop).copy()
    for c in ['Sex', 'Marital', 'Residence', 'His_BloodTrans', 'Soil_Contact', 'Eating_RawMeat', 'O_Training',
              'O_Contact_muscle_blood', 'O_Accident', 'O_Smoking_breaks', 'O_Washing_hand', 'E_Pets',
              'E_Treated_water', 'E_Tank_covered', 'E_Vacant_lots', 'E_Flooded_Area', 'E_Rodents']:
        X[c] = code12(X[c])
    X = X.rename(columns={'Soil_Contact': 'soil_contact', 'Eating_RawMeat': 'raw_meat', 'Age': 'age',
                          'E_Pets': 'pets_at_home', 'Sex': 'male_flag'})
    return X, y


def iran_suicide():
    import pyreadstat
    d, _ = pyreadstat.read_sav(fetch('iran_suicide'))
    y = (d['Toxo.IgG'] == 2).astype(int)        # 2 = positive (153 = 98 + 55 positives reported in the paper)
    X = pd.DataFrame({'suicide_attempt_group': (d['Group'] == 1).astype(float), 'sex_code': d['Geneus'],
                      'age_group': d['Age'], 'education': d['Education'], 'place_code': d['place'],
                      'raw_meat_code': d['rawmeat'], 'fastfood_code': d['fastfood'], 'cat_contact_code': d['contactcat'],
                      'history_suicide_code': d['Historyofsuicide'], 'mental_disorder_code': d['Historyofmentaldisorder'],
                      'income': d['income'], 'parents_education': d['Parentseducation']})
    # IL-6 / TNF / IFN-g are measured AFTER infection status is known and differ by infection -> not used
    return X, y


def philippines_cebu():
    d = pd.read_excel(fetch('philippines_cebu'), sheet_name=0)
    d['RST'] = pd.to_numeric(d['RST'], errors='coerce')
    d = d[d['RST'].isin([0, 1])]
    y = d['RST'].astype(int)
    cols = ['AGE', 'SEX', 'CIV', 'LOCQ', 'EDU', 'PROCAT', 'PRSM', 'PRVM', 'CAT', 'NUM', 'YEA', 'FRE', 'PO', 'PET', 'STR', 'PRE']
    X = d[cols].apply(pd.to_numeric, errors='coerce').rename(columns={'AGE': 'age', 'CAT': 'cat_owner', 'FRE': 'cat_contact_freq'})
    return X.reset_index(drop=True), y.reset_index(drop=True)


def thai_border_pregnant():
    d = pd.read_excel(fetch('thai_border_pregnant'))
    y = (d['Toxoplasmosis IgG'].astype(str).str.strip().str.lower() == 'pos').astype(int)
    keep = ['Age', 'Migrant/refugee', 'Gravidity', 'Parity', 'Primigravida', 'Poor obstetric outcome in history',
            'Miscarriage in history', 'Neonatal death in history', 'Preterm labor in history', 'Literate',
            'Helminths Hookworm', 'Helminths Ascaris', 'Helminths Trichuris', 'months at address', 'Ethnic group']
    X = d[keep].apply(pd.to_numeric, errors='coerce').rename(columns={'Age': 'age'})
    X = pd.concat([X, pd.get_dummies(d['Clinic'], prefix='clinic', dtype=float)], axis=1)
    return X, y


def brazil_homeless():
    d = pd.read_excel(fetch('brazil_homeless'))
    d.columns = [str(c).strip() for c in d.columns]
    target = [c for c in d.columns if 'T. gondii' in c][0]
    y = (d[target].astype(str).str.strip().str.lower() != 'negative').astype(int)
    X = d.drop(columns=[target, '1.Sample']).apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.notna().any()]
    X = X.rename(columns={'2.Age': 'age'})
    return X, y


def ghana_sickle_cell():
    d = pd.read_excel(fetch('ghana_sickle_cell'))
    d.columns = [str(c).strip() for c in d.columns]
    y = (d['ELISA'].astype(str).str.upper().str.strip() == 'POSITIVE').astype(int)
    occ = d['Occupation'].astype(str).str.lower().str.strip()
    X = pd.DataFrame({'age': d['Age(Raw) yrs'], 'male': yes(d['Gender'], ('male',)),
                      'education': d['Education'].astype(str).str.lower().str.strip().map({'none': 0, 'primary': 1, 'jhs': 2, 'shs': 3, 'tertiary': 4}),
                      'student': (occ == 'student').astype(float), 'married': yes(d['Marrital status'], ('married',)),
                      'rural': yes(d['Residence'], ('rural',)), 'blood_transfusion': yes(d['Blood Transfusion']),
                      'genotype_SS': (d['Genotype'] == 'SS').astype(float), 'cat_owner': yes(d['Cat Ownership']),
                      'undercooked_meat': yes(d['Eating Undercooked Meat'])})
    # RDT (rapid Toxoplasma test) is excluded: it is a Toxoplasma test = label leakage
    return X, y


LOADERS = {k: globals()[k] for k in SOURCES}


def load_all():
    out = {}
    for k, f in LOADERS.items():
        X, y = f()
        X = X.apply(pd.to_numeric, errors='coerce').astype(float)
        out[k] = (X, y.astype(int), SOURCES[k])
    return out

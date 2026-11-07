"""
Public European individual-level Toxoplasma datasets (CC BY), used together with the US (NHANES) data.

czech_slovak  Flegr lab, Figshare 1585139 (raw data of Epidemiol Infect 2017, doi 10.1017/S095026881700019X).
              Online survey, Czech + Slovak adults; '054_toxo' = result of an earlier serological test (1 neg, 2 pos);
              exposures: cat bites/scratches, cats/dogs at home, raw meat, soil, unwashed vegetables, water, STDs, residence.
france_donors Zenodo 18330122, French blood donors 2019 (1,995): age group, sex, region, Evolis (Bio-Rad) Toxoplasma IgG.
              Positive = IgG >= 10 IU/mL (clear valley in the bimodal distribution; 8-10 IU/mL: 8 donors).
"""
import os
import urllib.request

import numpy as np
import pandas as pd

EU = os.path.join('data', 'external', 'europe')
URLS = {'catEpidemiology7270.xlsx': 'https://ndownloader.figshare.com/files/2372261',
        'france_donors.xlsx': 'https://zenodo.org/records/18330122/files/Raw%20Data-Toxoplasma%20seroprevalence%20among%20blood%20donors.xlsx?download=1'}


def _get(name):
    p = os.path.join(EU, name)
    if not os.path.exists(p):
        os.makedirs(EU, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], p)
    return p


def czech_slovak():
    d = pd.read_excel(_get('catEpidemiology7270.xlsx'))
    d = d[d['054_toxo'].isin([1, 2])].reset_index(drop=True)
    y = (d['054_toxo'] == 2).astype(int)
    X = pd.DataFrame({
        'age': d['012_age'], 'male': d['013_sex'].map({1: 1.0, 2: 0.0}), 'residence_size': d['018_living'],
        'dogs': d['036_dogs'], 'dog_bitten': d['037_dogBitBin'], 'cats': d['043_cats'],
        'cat_bitten': d['045_catBitBin'], 'cat_scratched': d['046_catScratchBin'],
        'cat_love': d['catLoveOrd'], 'cat_touch': d['catToucheOrd'],
        'raw_meat': d['327_meat'], 'garden_soil': d['328_soil'], 'unwashed_vegetables': d['329_unwashed'],
        'rural_childhood': d['330_country'], 'untreated_water': d['331_watter'], 'sexual_risk': d['332_STD']})
    return X.astype(float), y


AGE_MID = {'18-29': 23.5, '30-39': 34.5, '40-49': 44.5, '50-59': 54.5, '60-69': 64.5, '>70': 72.0}


def france_donors():
    d = pd.read_excel(_get('france_donors.xlsx'))
    y = (d['EvolisToxoplasma IgG'] >= 10).astype(int)
    X = pd.DataFrame({'age': d['Age'].map(AGE_MID), 'male': (d['Sex'] == 'Male').astype(float)})
    X = pd.concat([X, pd.get_dummies(d['Region'], prefix='region', dtype=float)], axis=1)
    return X.astype(float), y

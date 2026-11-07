"""
Public US individual-level Toxoplasma datasets outside NHANES (found by the Europe PMC search, Oct 2026).

1. johns_hopkins_baltimore  PMC4363491, PLoS One 2015 (doi 10.1371/journal.pone.0116696)
   2,660 Ashkenazi Jewish adults (schizophrenia, bipolar, relatives, controls) recruited 1996-2011 by Johns Hopkins.
   Plasma IgG to T. gondii, CMV, HSV-1, HHV-6, gliadin and CRP, all given as plate-standardised z-scores.
   No exposure questionnaire. The z-score is clearly bimodal; positive = z > 1.75 (the valley), 20.5% positive.
2. iowa_beach_saliva        PMC8199218, IJERPH 2021: salivary antibody MFI only, no questionnaire (not modelled).
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

EXT = os.path.join('data', 'external')


def johns_hopkins_baltimore():
    path = os.path.join(EXT, 'johns_hopkins_baltimore', 'pone.0116696.s001.txt')
    if not os.path.exists(path):
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get('https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4363491/supplementaryFiles', timeout=300)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        open(path, 'wb').write(z.read('pone.0116696.s001.txt'))
    d = pd.read_csv(path, sep=None, engine='python')
    y = (d['toxoigg_z_r1'] > 1.75).astype(int)
    X = pd.DataFrame({'age': d['age_at_plasma'], 'male': (d['sex'] == 'M').astype(float),
                      'schizophrenia_or_bipolar': d['status'].isin(['Sz/Sza', 'Bipolar']).astype(float),
                      'cmv_igg_z': d['cmvigg_z_r1'], 'hsv1_igg_z': d['hsv1igg_z_r1'], 'hhv6_igg_z': d['hhv6igg_z_r1'],
                      'crp_z': d['crp_79102_z_r1'], 'gliadin_igg_z': d['gliadinigg_z_r1'],
                      'storage_years': d['storageYears']})
    return X, y

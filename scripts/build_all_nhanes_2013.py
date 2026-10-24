"""
Merge EVERY one-row-per-person NHANES 2013-2014 file onto the adult Toxoplasma cohort.

Purpose: an exhaustive test of whether ANY variable NHANES measured in 2013-2014 can raise the
prediction score. Excluded:
  * SSTOXO_H (the Toxoplasma test itself: IgG titer, IgM, avidity -> label leakage)
  * long-format files with many rows per person (food items, supplements, prescriptions,
    minute-level activity) and pooled-sample files without SEQN
  * survey weights, PSU/strata, interviewer IDs and comment codes (not person characteristics)
Output: data/processed/nhanes_2013_all_variables.parquet (or .csv.gz)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

RAW = os.path.join('data', 'raw', 'NHANES_all')
BASE_CSV = os.path.join('data', 'processed', 'nhanes_adult_toxo_clean.csv')
OUT = os.path.join('data', 'processed', 'nhanes_2013_all_variables.csv.gz')
CDC_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/{}.xpt'

LEAK_FILES = {'SSTOXO_H'}
DROP_PATTERNS = [r'^WT', r'^SDMV', r'^SDDSRVYR$', r'EXMER$', r'LC$', r'^RIDSTATR$']


def build(seqns):
    frames, log = [], []
    for path in sorted(glob.glob(os.path.join(RAW, '*.xpt'))):
        name = os.path.basename(path)[:-4]
        if name in LEAK_FILES:
            continue
        try:
            d = pd.read_sas(path, format='xport')
        except Exception as e:  # noqa: BLE001
            log.append((name, 'unreadable', str(e)[:60]))
            continue
        if 'SEQN' not in d or d['SEQN'].duplicated().any():
            log.append((name, 'skipped (not one row per person)', len(d)))
            continue
        d = d[d['SEQN'].isin(seqns)]
        keep = [c for c in d.columns if c != 'SEQN'
                and pd.api.types.is_numeric_dtype(d[c])
                and not any(re.search(p, c) for p in DROP_PATTERNS)]
        d = d[['SEQN'] + keep].copy()
        for c in keep:   # XPT zero artifact
            tiny = (d[c].abs() > 0) & (d[c].abs() < 1e-70)
            d.loc[tiny, c] = 0.0
        d = d.rename(columns={c: f'{name.replace("_H", "")}__{c}' for c in keep})
        frames.append(d.set_index('SEQN'))
        log.append((name, 'merged', len(keep)))
    wide = pd.concat(frames, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]
    # drop columns with < 5% non-missing values or a single value
    nonmiss = wide.notna().mean()
    wide = wide.loc[:, (nonmiss >= 0.05) & (wide.nunique() > 1)]
    return wide, pd.DataFrame(log, columns=['file', 'status', 'info'])


def main():
    base = pd.read_csv(BASE_CSV)
    wide, log = build(set(base['SEQN']))
    out = base[['SEQN', 'target_toxo_pos']].set_index('SEQN').join(wide).reset_index()
    out.to_csv(OUT, index=False, compression='gzip')
    print(f'Saved {OUT}: {out.shape[0]} people x {out.shape[1] - 2} variables '
          f'from {int((log.status == "merged").sum())} files')
    return out, log


if __name__ == '__main__':
    main()

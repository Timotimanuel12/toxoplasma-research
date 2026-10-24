"""Read NHANES III (1988-1994) fixed-width .dat files using the CDC SAS layout (.sas) files."""
import re
import pandas as pd


def layout(sas_path):
    txt = open(sas_path, encoding='latin1').read()
    inp = txt[txt.upper().index('INPUT'):]
    pos = {}
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)\s+(\d+)(?:-(\d+))?\s*$', inp, flags=re.M):
        name, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        pos.setdefault(name, (a - 1, b))
    labels = dict(re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*"([^"]*)"', txt, flags=re.M))
    return pos, labels


def read(dat_path, sas_path, cols):
    pos, _ = layout(sas_path)
    specs = [pos[c] for c in cols]
    return pd.read_fwf(dat_path, colspecs=specs, names=cols, header=None, na_values=['.', ''],
                       encoding='latin1')

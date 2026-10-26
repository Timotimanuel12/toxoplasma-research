"""Compare the public Makassar file with the numbers printed in Polanunu et al. (2021)."""
import sys

import numpy as np
import pandas as pd
import pyreadstat

sys.stdout.reconfigure(encoding='utf-8')
raw, meta = pyreadstat.read_sav(sys.argv[1])
clean = pd.read_csv(sys.argv[2])


def check(name, ours, paper):
    ok = 'OK ' if str(ours) == str(paper) else 'DIFF'
    print(f'  [{ok}] {name:45} data: {ours!s:14} paper: {paper}')


print('=== Table 1 (all 184)')
check('Age mean (SD)', f'{raw.age.mean():.2f} ({raw.age.std():.2f})', '26.97 (5.63)')
check('Primigravida n', int((raw.Gravidity == 1).sum()), 62)
check('Abortion history n', int((raw.AbortusHistory == 1).sum()), 32)
for code, n in zip([1, 2, 3, 4], [24, 31, 87, 42]):
    check(f'Education {meta.variable_value_labels["Education"][code]}', int((raw.Education == code).sum()), n)
check('Employed n', int((raw.Employed == 1).sum()), 32)
check('IgM index range (median)', f'{raw.ToxoIgM_Index.min():.2f}-{raw.ToxoIgM_Index.max():.2f} ({raw.ToxoIgM_Index.median():.2f})', '0.06-1.02 (0.22)')
check('IgG index range (median)', f'{raw.ToxoIgG_Index.min():.2f}-{raw.ToxoIgG_Index.max():.2f} ({raw.ToxoIgG_Index.median():.2f})', '0.09-3.00 (0.33)')
check('IgM neg / pos / borderline', '/'.join(str(int((raw.ToxoIgM_Positivity == c).sum())) for c in [2, 1, 3]), '178/0/6')
check('IgG neg / pos / borderline', '/'.join(str(int((raw.ToxoIgG_Positivity == c).sum())) for c in [2, 1, 3]), '122/60/2')

print('\n=== Positivity cut-offs (paper: negative <0.9, borderline 0.9-1.1, positive >1.1)')
for c, lab in [(2, 'negative'), (3, 'borderline'), (1, 'positive')]:
    s = raw.loc[raw.ToxoIgG_Positivity == c, 'ToxoIgG_Index']
    print(f'  IgG {lab:10} index {s.min():.3f}-{s.max():.3f}')

print('\n=== Table 2 (sites; ID prefix -> health centre)')
paper_sites = {'SD': ('Sudiang', 22, 36.36), 'SR': ('Sudiang Raya', 5, 40.00), 'PC': ('Paccerakkang', 10, 50.00),
               'KB': ('Kaluku Bodoa', 37, 29.72), 'AT': ('Antang', 37, 27.02), 'KK': ('Kassi-Kassi', 25, 32.00),
               'MG': ('Mangasa', 10, 50.00), 'JG': ('Jongaya', 16, 31.25), 'TT': ('Tamalate', 22, 27.27)}
for code, (name, n, pct) in paper_sites.items():
    s = clean[clean.site == code]
    bl = int(raw.loc[raw.Code.str.startswith(code), 'ToxoIgG_Positivity'].eq(3).sum())
    check(f'{code} = {name}: n, % positive', f'{len(s)}, {100 * s.igg_positive.mean():.2f}' + (f' (+{bl} borderline)' if bl else ''), f'{n}, {pct:.2f}')

print('\n=== Table 3 (182 women, borderline excluded): positive / negative counts')
t3 = raw[raw.ToxoIgG_Positivity != 3].copy()
t3['y'] = (t3.ToxoIgG_Positivity == 1).astype(int)
rows = [('Multigravida', t3.Gravidity > 1, (42, 77)), ('Primigravida', t3.Gravidity == 1, (18, 45)),
        ('Miscarriage yes', t3.AbortusHistory == 1, (11, 19)), ('Education <=9 y (elem+junior)', t3.Education <= 2, (15, 38)),
        ('Employed yes', t3.Employed == 1, (14, 23)), ('Washing un-chlorinated', t3.WaterForWashing == 1, (33, 57)),
        ('Cooking untreated', t3.WaterForCookingPreparation == 1, (22, 40)),
        ('Drinking "filtered" (= gallon water in file)', t3.WaterForDrinking == 1, (53, 84)),
        ('Raising cat', t3.RaisingACat == 1, (11, 13)), ('Contact with cat', t3.ContactWithCat == 1, (35, 14)),
        ('Cleaning litter', t3.CleaningLittebox == 1, (11, 7)), ('Contact with soil', t3.ContactSoil2 == 1, (12, 13)),
        ('Undercooked meat (chicken satay)', t3.ConsumeChickenSatay == 1, (50, 44)),
        ('Raw vegetables', t3.ConsumeRawVegetable == 1, (37, 62)), ('Unboiled/filtered water', t3.ConsumeUnboiledWater == 1, (55, 67))]
for name, mask, (pp, pn) in rows:
    check(name, f'{int(t3.y[mask].sum())}/{int((1 - t3.y[mask]).sum())}', f'{pp}/{pn}')

print('\n=== Multivariable model (paper: variables with univariate p<0.1, 182 women)')
X = pd.DataFrame({'drinking filtered': t3.WaterForDrinking == 1, 'contact with cat': t3.ContactWithCat == 1,
                  'cleaning litter': t3.CleaningLittebox == 1, 'contact with soil': t3.ContactSoil2 == 1,
                  'undercooked meat': t3.ConsumeChickenSatay == 1, 'unboiled water': t3.ConsumeUnboiledWater == 1}).astype(float)
Xm = np.column_stack([np.ones(len(X)), X.to_numpy()])
yv = t3.y.to_numpy(float)
b = np.zeros(Xm.shape[1])
for _ in range(100):
    p = 1 / (1 + np.exp(-Xm @ b))
    step = np.linalg.solve(Xm.T @ (Xm * (p * (1 - p))[:, None]), Xm.T @ (yv - p))
    b += step
    if np.abs(step).max() < 1e-10:
        break
p = 1 / (1 + np.exp(-Xm @ b))
se = np.sqrt(np.diag(np.linalg.inv(Xm.T @ (Xm * (p * (1 - p))[:, None]))))
paper_aor = {'drinking filtered': '1.49 (0.42-5.31)', 'contact with cat': '10.45 (3.77-28.99)', 'cleaning litter': '0.78 (0.20-3.03)',
             'contact with soil': '0.84 (0.28-2.53)', 'undercooked meat': '9.72 (3.71-25.48)', 'unboiled water': '5.98 (1.77-20.23)'}
for j, name in enumerate(X.columns, start=1):
    check(f'AOR {name}', f'{np.exp(b[j]):.2f} ({np.exp(b[j] - 1.96 * se[j]):.2f}-{np.exp(b[j] + 1.96 * se[j]):.2f})', paper_aor[name])

print('\n=== Variables in the paper but not in the public file')
print('  House member (>4 vs <=4):', 'present' if any('house' in c.lower() or 'member' in c.lower() for c in raw.columns) else 'NOT in the file')

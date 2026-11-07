"""Figures for the IEEE paper (standard machine-learning models only).
Run from the repository root after paper/compute_paper_results.py:  python paper/make_figures.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, 'scripts')
import build_bio_features as bbf  # noqa: E402

RES = os.path.join('paper', 'results')
OUT = os.path.join('paper', 'figures')
os.makedirs(OUT, exist_ok=True)
INK2, GRID = '#555555', '#dddddd'
BLUE, ORANGE, GREEN, GREY = '#1f5fa8', '#d9622b', '#3f8f5a', '#9a9a9a'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
W1 = 3.45   # IEEE single-column width (inches)

# ---------------------------------------------------------------- Fig. 2: models x feature sets
res = pd.read_csv(os.path.join(RES, 'models_by_featureset.csv'))
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
sets = ['Baseline', 'Baseline + lifestyle', 'Baseline + lifestyle + 44 markers']
setcol = dict(zip(sets, [GREY, BLUE, ORANGE]))
setlab = dict(zip(sets, ['Baseline', '+ lifestyle', '+ lifestyle + 44 markers']))
fig, axes = plt.subplots(2, 1, figsize=(W1, 3.1), sharex=True)
for ax, col, title in [(axes[0], 'cv_mean', '(a) Cross-validated ROC-AUC (training set, 15 folds)'),
                       (axes[1], 'test_auc', '(b) Test ROC-AUC (1,002 held-out adults)')]:
    for j, s in enumerate(sets):
        r = res[res.set == s].set_index('model').loc[models]
        x = np.arange(len(models)) + (j - 1) * 0.22
        if col == 'cv_mean':
            ax.errorbar(x, r[col], yerr=r['cv_sd'], fmt='o', ms=4, color=setcol[s], elinewidth=0.8, capsize=0, label=setlab[s])
        else:
            ax.plot(x, r[col], 's', ms=4, color=setcol[s], label=setlab[s])
    ax.set_title(title, loc='left'); ax.set_ylabel('ROC-AUC'); ax.grid(axis='x', visible=False)
axes[0].set_ylim(0.67, 0.75); axes[1].set_ylim(0.68, 0.74)
axes[1].set_xticks(range(len(models))); axes[1].set_xticklabels(['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'XGBoost'])
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=3, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(h_pad=0.6)
fig.savefig(os.path.join(OUT, 'fig2_models.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: adjusted odds ratios
def logit_fit(X, y, iters=60):
    X = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        s = np.linalg.solve(H, X.T @ (y - p)); b += s
        if np.abs(s).max() < 1e-9:
            break
    return b[1:], np.sqrt(np.diag(np.linalg.inv(H)))[1:]


df = pd.read_csv(bbf.OUT_CSV)
y = df['target_toxo_pos'].to_numpy(float)
adj = np.column_stack([df['RIDAGEYR'] / 10, df['RIAGENDR'] == 1, df['DMDBORN4'] == 2]).astype(float)
LABEL = {'toxocara_mfi_log': 'Toxocara IgG level (per SD)', 'spanish_interview': 'Spanish-language interview',
         'no_health_insurance': 'No health insurance', 'crowding': 'Crowding, persons/room (per SD)',
         'missing_teeth': 'Missing teeth (per SD)', 'household_size': 'Household size (per SD)',
         'food_insecurity': 'Food insecurity (per SD)', 'not_us_citizen': 'Not a US citizen', 'hsv2': 'HSV-2 IgG',
         'blood_mercury_log': 'Blood mercury (per SD)', 'shellfish_30d': 'Shellfish, past 30 d',
         'blood_lead_log': 'Blood lead (per SD)', 'cotinine_log': 'Serum cotinine (per SD)', 'hsv1': 'HSV-1 IgG',
         'work_construction_soil': 'Construction/soil work', 'eosinophil_pct': 'Eosinophils % (per SD)',
         'home_rented': 'Rented home'}
rows = []
for c in bbf.NEW_FEATURES:
    x = df[c]; binary = set(x.dropna().unique()) <= {0.0, 1.0}
    xs = x if binary else (x - x.mean()) / x.std(); m = x.notna().to_numpy()
    b, se = logit_fit(np.column_stack([xs[m], adj[m]]), y[m])
    rows.append((c, np.exp(b[0]), np.exp(b[0] - 1.96 * se[0]), np.exp(b[0] + 1.96 * se[0]), 2 * stats.norm.sf(abs(b[0] / se[0]))))
orr = pd.DataFrame(rows, columns=['f', 'or', 'lo', 'hi', 'p'])
orr.to_csv(os.path.join(RES, 'adjusted_odds_ratios.csv'), index=False)
sig = orr[orr.p < 0.05].sort_values('or').reset_index(drop=True)
assert len(sig) == 17, len(sig)
fig, ax = plt.subplots(figsize=(W1, 2.9))
yy = np.arange(len(sig)); col = [BLUE if o > 1 else ORANGE for o in sig['or']]
ax.hlines(yy, sig['lo'], sig['hi'], color=col, lw=1.1)
ax.scatter(sig['or'], yy, color=col, s=12, zorder=3)
ax.axvline(1, color=INK2, lw=0.7, ls='--')
ax.set_xscale('log'); ticks = [0.6, 0.8, 1, 1.5, 2, 3, 4]
ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
ax.set_yticks(yy); ax.set_yticklabels([LABEL[f] for f in sig['f']])
ax.set_xlabel('Odds ratio adjusted for age, sex and birthplace (log scale)')
ax.grid(axis='y', visible=False)
fig.savefig(os.path.join(OUT, 'fig3_adjusted_or.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: ROC curves on the test set
cur = json.load(open(os.path.join(RES, 'roc_curves.json')))
lines = [('Logistic Regression|Baseline', 'Logistic regression, baseline', GREY, '-'),
         ('XGBoost|Baseline + lifestyle + 44 markers', 'XGBoost + 44 markers', GREEN, '-'),
         ('Logistic Regression|Baseline + lifestyle + 44 markers', 'Logistic regression + 44 markers', BLUE, '-'),
         ('STEP|5 + XGBoost ensemble', 'Improved model (pooled, ensemble)', ORANGE, '-')]
fig, ax = plt.subplots(figsize=(W1, 2.7))
for k, lab, c, ls in lines:
    d = cur[k]; ax.plot(d['fpr'], d['tpr'], color=c, ls=ls, lw=1.2, label=f"{lab} ({d['auc']:.3f})")
ax.plot([0, 1], [0, 1], color=GRID, ls='--', lw=0.8)
ax.set_xlabel('False positive rate (1 - specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
ax.legend(loc='lower right', handlelength=1.4)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
fig.savefig(os.path.join(OUT, 'fig4_roc.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: improvement steps + training harder
lad = pd.read_csv(os.path.join(RES, 'improvement_steps.csv'))
harder = json.load(open(os.path.join('data', 'processed', 'train_harder_results.json')))['harder']
fig, axes = plt.subplots(1, 2, figsize=(W1, 1.75), gridspec_kw={'width_ratios': [1.1, 1]})
ax = axes[0]; x = np.arange(len(lad))
ax.errorbar(x - 0.1, lad['CV ROC-AUC (2013-14 people)'], yerr=lad['CV sd'], fmt='o', ms=3.5, color=BLUE,
            elinewidth=0.8, capsize=0, label='CV')
ax.plot(x + 0.1, lad['TEST ROC-AUC'], 's', ms=3.5, color=ORANGE, label='Test')
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(len(lad))]); ax.set_xlabel('Step'); ax.set_ylabel('ROC-AUC')
ax.set_ylim(0.675, 0.765); ax.set_title('(a) Improvement steps', loc='left')
ax.legend(loc='lower right', ncol=2, handletextpad=0.2, borderaxespad=0.1, columnspacing=0.6)
ax.grid(axis='x', visible=False)
ax = axes[1]
lab = [f"{h['max_depth']}/{h['trees'] // 100 / 10:g}k" for h in harder]
ax.plot(range(len(harder)), [h['train AUC'] for h in harder], 'o-', ms=3.5, lw=1, color=GREY, label='Training')
ax.plot(range(len(harder)), [h['validation AUC (2013-14)'] for h in harder], 's-', ms=3.5, lw=1, color=BLUE, label='Unseen')
ax.set_xticks(range(len(harder))); ax.set_xticklabels(lab, rotation=45)
ax.set_xlabel('Depth / trees'); ax.set_ylim(0.68, 1.02); ax.set_title('(b) Training harder', loc='left')
ax.legend(loc='center right', handletextpad=0.2, borderaxespad=0.1)
ax.grid(axis='x', visible=False)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(OUT, 'fig5_steps.pdf'))
plt.close(fig)
print('figures written to', OUT)

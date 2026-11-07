# 08_zenodo: Zenodo (CERN)

Zenodo API search (2,136 Toxoplasma records, 25 per page); Dryad mirrors are filed under 09_dryad.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| france_blood_donors | France | Blood donors | 1995 | 1008 | 15 | 0.747 | 0.734 | 0.740 | **0.747** |
| medical_students_latam | Latin America (Spanish-language) | Medical students; questionnaire on cats, meat, hand-washing, water | 640 | 221 | 16 | 0.767 | 0.784 | 0.780 | **0.784** |
| rennes_congenital | France | Infants born to mothers with toxoplasmosis in pregnancy | 1171 | 126 | 1 | 0.550 | 0.601 | 0.610 | **0.610** |

## Datasets

- **france_blood_donors** (10.5281/zenodo.18330122): Toxoplasma seroprevalence among blood donors in France (2019) — Evolis IgG >= 10 IU/mL; only age group, sex and region available.
- **medical_students_latam** (10.5281/zenodo.23744904): Toxoplasmosis study database in medical students (Base de datos ... estudiantes de medicina)
- **rennes_congenital** (10.5281/zenodo.15490613): Anti-Toxoplasma IgA analysis, CHU de Rennes: infants of mothers infected in pregnancy — Different outcome: congenital transmission. All infant/placenta tests removed; only timing of maternal infection and age at sampling remain.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `08_zenodo.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/08_zenodo/08_zenodo.ipynb`

Folder layout: `08_zenodo.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

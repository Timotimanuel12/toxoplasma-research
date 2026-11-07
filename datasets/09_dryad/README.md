# 09_dryad: Dryad

Dryad API search (38 Toxoplasma records, almost all animal or laboratory); files from the Zenodo mirror.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| usa_students_risky_business | USA | University students, saliva IgG test | 1293 | 325 | 3 | 0.571 | 0.508 | 0.534 | **0.571** |
| usa_entrepreneur_event | USA | Professionals attending entrepreneurship events | 197 | 18 | 2 |  |  |  |  |

## Datasets

- **usa_students_risky_business** (10.5061/dryad.gd19rr3): Risky business: Toxoplasma infection and entrepreneurship behaviours (students) — Only 5 inputs: business major, sex, GPA and two career-interest items.
- **usa_entrepreneur_event** (10.5061/dryad.gd19rr3): Risky business: Toxoplasma status of professionals at entrepreneurship events — 18 positives: below the 20-case minimum, catalogued only.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `09_dryad.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/09_dryad/09_dryad.ipynb`

Folder layout: `09_dryad.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

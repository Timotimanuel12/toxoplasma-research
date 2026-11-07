# 03_bmc_springer_nature: BMC / Springer Nature journals (BMC Infectious Diseases, Parasites & Vectors, ...)

Participant-level supplementary files ("MOESM"), found through Europe PMC; Springer Nature also mirrors them on Figshare.

**How the journal datasets were found.** Europe PMC was searched for every open-access paper with supplementary files whose title/abstract mentions Toxoplasma, toxoplasmosis, gondii, TORCH, multiplex serology or pathogen burden (3,482 papers). Their data files (xlsx, xls, csv, sav, dta, ...) were downloaded from the PMC Open Access S3 bucket (1,407 papers had data files, 6.6 GB) and every table with >= 80 rows was scanned for a Toxoplasma result column (607 tables, 285 papers). Human, participant-level tables with at least two IgG classes were then checked by hand. Scripts: `datasets/_harvest/`.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| iran_suicide_attempters | Iran | Suicide attempters and controls | 500 | 153 | 12 | 0.632 | 0.631 | 0.625 | **0.632** |
| colombia_thyroid | Colombia | Thyroid disease cases and controls | 704 | 59 | 5 | 0.538 | 0.484 | 0.483 | **0.538** |
| mexico_liver_disease | Mexico | Liver disease patients and controls | 156 | 16 | 54 |  |  |  |  |

## Datasets

- **iran_suicide_attempters** (10.1186/s12879-025-11200-1): Epidemiological aspects and cytokine profiles of T. gondii in suicide attempters, Lorestan
- **colombia_thyroid** (10.1186/s12879-019-4450-0): Association between T. gondii infection and thyroid dysfunction: case-control study
- **mexico_liver_disease** (10.1186/1756-3305-4-75): T. gondii infection and liver disease: case-control study, northern Mexico — 16 positives: below the 20-case minimum, catalogued only.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `03_bmc_springer_nature.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/03_bmc_springer_nature/03_bmc_springer_nature.ipynb`

Folder layout: `03_bmc_springer_nature.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

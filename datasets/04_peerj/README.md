# 04_peerj: PeerJ

PeerJ publishes the raw data of every article as supplementary files; found through Europe PMC.

**How the journal datasets were found.** Europe PMC was searched for every open-access paper with supplementary files whose title/abstract mentions Toxoplasma, toxoplasmosis, gondii, TORCH, multiplex serology or pathogen burden (3,482 papers). Their data files (xlsx, xls, csv, sav, dta, ...) were downloaded from the PMC Open Access S3 bucket (1,407 papers had data files, 6.6 GB) and every table with >= 80 rows was scanned for a Toxoplasma result column (607 tables, 285 papers). Human, participant-level tables with at least two IgG classes were then checked by hand. Scripts: `datasets/_harvest/`.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| china_gynaecological_tumour | China | Women with gynaecological tumours + matched controls | 920 | 69 | 12 | 0.839 | 0.892 | 0.877 | **0.892** |
| china_leukemia_children | China | Children with leukaemia + matched controls | 678 | 79 | 11 | 0.544 | 0.584 | 0.588 | **0.588** |

## Datasets

- **china_gynaecological_tumour** (10.7717/peerj.14569): T. gondii in women with a gynaecological tumour and controls, eastern China
- **china_leukemia_children** (10.7717/peerj.6604): Seroprevalence and risk factors of T. gondii in children with leukaemia, Shandong — Two sheets (patients / controls) harmonised; direct exposures: cats, dogs, pigs, raw meat, raw vegetables, soil, water.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `04_peerj.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/04_peerj/04_peerj.ipynb`

Folder layout: `04_peerj.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

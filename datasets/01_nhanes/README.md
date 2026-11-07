# 01_nhanes: CDC NHANES (U.S. National Health and Nutrition Examination Survey)

Public, de-identified survey files from CDC/NCHS (wwwn.cdc.gov/nchs/nhanes), downloaded as .XPT by the build scripts. Four Toxoplasma cohorts with three different assays, so they are evaluated separately, never pooled across assays. New in this folder: NHANES 1999-2004 (files LAB17, L17_B, L17_C; IgG tested at ages 6-49).

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| nhanes_1999_2004 | USA | Adults 18-49, 1999-2004, CDC EIA IgG >= 7 IU/mL; demographics, housing, insurance, smoking | 8485 | 1235 | 15 | 0.720 | 0.715 | 0.718 | **0.720** |
| nhanes_2013_2014_markers | USA | Adults 18+, 2013-2014, Bio-Rad IgG >= 33 IU/mL; baseline + lifestyle + 44 markers (the paper) | 5006 | 750 | 54 | 0.699 | 0.718 | 0.713 | **0.718** |
| nhanes_2009_2014_pooled | USA | Adults 18+, 2009-2014 pooled, same assay; baseline + markers | 14649 | 2495 | 58 | 0.726 | 0.738 | 0.742 | **0.742** |
| nhanes_iii_1988_1994 | USA | Adults 18+, 1988-1994; includes cat, garden, meat-frequency, region and water questions | 15524 | 4263 | 37 | 0.691 | 0.706 | 0.703 | **0.706** |

## Datasets

- **nhanes_1999_2004**: Adults 18-49, 1999-2004, CDC EIA IgG >= 7 IU/mL; demographics, housing, insurance, smoking
- **nhanes_2013_2014_markers**: Adults 18+, 2013-2014, Bio-Rad IgG >= 33 IU/mL; baseline + lifestyle + 44 markers (the paper)
- **nhanes_2009_2014_pooled**: Adults 18+, 2009-2014 pooled, same assay; baseline + markers
- **nhanes_iii_1988_1994**: Adults 18+, 1988-1994; includes cat, garden, meat-frequency, region and water questions

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `01_nhanes.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/01_nhanes/01_nhanes.ipynb`

Folder layout: `01_nhanes.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

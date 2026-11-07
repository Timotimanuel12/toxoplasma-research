# 10_harvard_dataverse: Harvard Dataverse

Dataverse search API (175 hits); NHANES copies hosted by ICPSR excluded (they belong to 01_nhanes).

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| ghana_ashanti_pregnant | Ghana | Pregnant women attending antenatal clinics | 300 | 150 | 18 | 0.672 | 0.650 | 0.608 | **0.672** |

## Datasets

- **ghana_ashanti_pregnant** (10.7910/DVN/HXJONT): Seroprevalence, risk factors and haematological impact of Toxoplasma infection, Ashanti Region — Direct exposures: cat ownership, cat litter, handling raw meat, meat preference, cooking, raw vegetables, drinking water; plus full blood count.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `10_harvard_dataverse.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/10_harvard_dataverse/10_harvard_dataverse.ipynb`

Folder layout: `10_harvard_dataverse.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

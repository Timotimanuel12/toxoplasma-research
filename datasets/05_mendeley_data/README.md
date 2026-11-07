# 05_mendeley_data: Mendeley Data (Elsevier data repository)

Searched with the public API (data.mendeley.com/api/research-data/search, 76 Toxoplasma records); files downloaded through public-api links. MDPI journals were also checked and had no usable participant-level data.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| nigeria_pregnant | Nigeria | Pregnant women at antenatal clinics | 250 | 171 | 43 | 0.552 | 0.558 | 0.533 | **0.558** |
| mexico_testosterone | Mexico | Adults, serologically tested | 213 | 35 | 15 | 0.647 | 0.652 | 0.670 | **0.670** |
| thailand_bangphae_children | Thailand | Healthy children and young adults, paired sera 2012 and 2015 | 573 | 24 | 5 | 0.615 | 0.580 | 0.585 | **0.615** |
| japan_uveitis_patients | Japan | Ophthalmology patients tested for Toxoplasma IgG | 83 | 17 | 4 |  |  |  |  |

## Datasets

- **nigeria_pregnant** (10.17632/s84shxhhfv.1): Toxoplasmosis among pregnant women in Nigeria (Dr Onyinye study) — Rich direct exposures: cat possession / contact / litter / proximity, stray cats, gardening, farm work, undercooked and raw meat, untreated water, unpasteurised milk, water sources, HIV status. Baby outcomes removed.
- **mexico_testosterone** (10.17632/dxg3498c7y.1): Toxoplasma infection, testosterone and psychopathological symptoms — Inputs: age, weight, height, BMI, sex, testosterone and SCL-90 symptom scales (possible consequences).
- **thailand_bangphae_children** (10.17632/mcw665mzrp.1): T. gondii seroprevalence by Sabin-Feldman dye test in children and young adults, Bangphae — Label = 2015 dye test; the 2012 result is removed (it measures the same infection).
- **japan_uveitis_patients** (10.17632/m4hvf5g7mj.1): Low Toxoplasma IgG seroprevalence in younger patients: ocular toxoplasmosis diagnosis — 17 positives: below the 20-case minimum, catalogued only.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `05_mendeley_data.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/05_mendeley_data/05_mendeley_data.ipynb`

Folder layout: `05_mendeley_data.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

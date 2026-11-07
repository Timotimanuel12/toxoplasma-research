# 06_other_journals: Other journals (ASTMH, Frontiers, Elsevier Heliyon, Wiley/Hindawi)

Publishers that contribute one or two datasets each; found through Europe PMC.

**How the journal datasets were found.** Europe PMC was searched for every open-access paper with supplementary files whose title/abstract mentions Toxoplasma, toxoplasmosis, gondii, TORCH, multiplex serology or pathogen burden (3,482 papers). Their data files (xlsx, xls, csv, sav, dta, ...) were downloaded from the PMC Open Access S3 bucket (1,407 papers had data files, 6.6 GB) and every table with >= 80 rows was scanned for a Toxoplasma result column (607 tables, 285 papers). Human, participant-level tables with at least two IgG classes were then checked by hand. Scripts: `datasets/_harvest/`.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| thai_myanmar_border_pregnant | Thailand/Myanmar | Refugee and migrant pregnant women | 199 | 62 | 18 | 0.506 | 0.584 | 0.532 | **0.584** |
| brazil_homeless | Brazil | Homeless adults in shelters | 120 | 43 | 43 | 0.510 | 0.378 | 0.390 | **0.510** |
| colombia_quindio_ocular | Colombia | Community survey with fundoscopy | 161 | 112 | 17 | 0.651 | 0.621 | 0.590 | **0.651** |
| iran_psychiatric_inpatients | Iran | Psychiatric inpatients + healthy controls | 200 | 86 | 43 | 0.683 | 0.702 | 0.702 | **0.702** |

## Datasets

- **thai_myanmar_border_pregnant** (10.4269/ajtmh.16-0999): T. gondii in refugee and migrant pregnant women, Thailand-Myanmar border — Am J Trop Med Hyg.
- **brazil_homeless** (10.3389/fpubh.2020.580637): Serosurvey of anti-T. gondii antibodies in homeless persons of São Paulo — Frontiers in Public Health.
- **colombia_quindio_ocular** (10.1016/j.heliyon.2021.e06659): High frequency of ocular toxoplasmosis in Quindío and related risk factors — Heliyon (Elsevier). Water, undercooked meat, cats, unwashed vegetables, education, SISBEN poverty level.
- **iran_psychiatric_inpatients** (PMC13662829 (J Parasitol Res, 2026)): Seroepidemiology of T. gondii in psychiatric inpatients and healthy controls, north Iran — Journal of Parasitology Research (Wiley/Hindawi). Cat and soil contact, raw vegetables, vegetable washing + blood count.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `06_other_journals.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/06_other_journals/06_other_journals.ipynb`

Folder layout: `06_other_journals.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

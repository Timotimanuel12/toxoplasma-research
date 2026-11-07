# 07_figshare: Figshare

Datasets deposited directly by authors (2,286 search hits; journal mirrors are filed under their journal). Most usable ones come from the Flegr lab, Charles University, Prague.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| czech_slovak_cat_survey | Czechia/Slovakia | Online survey, adults with a past Toxoplasma test | 1371 | 368 | 16 | 0.636 | 0.659 | 0.654 | **0.659** |
| czech_personality_borrelia | Czechia | Online survey (Facebook "Guinea pigs" community) | 962 | 210 | 26 | 0.634 | 0.607 | 0.616 | **0.634** |
| czech_health_stress | Czechia | Online survey | 747 | 153 | 8 | 0.652 | 0.612 | 0.601 | **0.652** |
| czech_students_toxo_cmv_iq | Czechia | Healthy university students, serologically tested | 556 | 97 | 29 | 0.560 | 0.533 | 0.587 | **0.587** |
| czech_toxo_cmv_longitudinal | Czechia | Adults re-tested after several years | 93 | 25 | 77 | 0.396 | 0.406 | 0.365 | **0.406** |
| czech_sexual_partners | Czechia | Couples tested together; exposure questionnaire | 1433 | 357 | 9 | 0.579 | 0.561 | 0.553 | **0.579** |
| poland_forensic_alcohol | Poland | Forensic autopsy cases | 535 | 248 | 10 | 0.547 | 0.593 | 0.548 | **0.593** |

## Datasets

- **czech_slovak_cat_survey** (10.6084/m9.figshare.1585139): Cats, dogs, raw meat and Toxoplasma: online survey of Czech and Slovak adults (Flegr lab) — Self-reported result of an earlier serological test; direct exposures (cats, raw meat, soil, water, unwashed vegetables).
- **czech_personality_borrelia** (10.6084/m9.figshare.21151075): Past Toxoplasma and Borrelia infections, personality and cognition (Flegr lab) — toxo: 2 = tested negative, 3 = tested positive (self-reported). Features are mostly personality / cognition (possible consequences).
- **czech_health_stress** (10.6084/m9.figshare.25188788): Toxoplasma and physical health, stress and anxiety (stress-coping hypothesis, Flegr lab) — Toxo: 2 = tested negative, 3 = tested positive (self-reported).
- **czech_students_toxo_cmv_iq** (10.6084/m9.figshare.21865677): Toxoplasma and CMV vs intelligence and memory in university students — Toxoplasma IgG measured in the lab; CMV IgG and Rh factor kept as biomarkers; cognitive scores are outcomes.
- **czech_toxo_cmv_longitudinal** (10.6084/m9.figshare.32706477): Toxoplasma and CMV: longitudinal study of cognitive change
- **czech_sexual_partners** (10.6084/m9.figshare.9198008): Male-to-female presumed transmission of toxoplasmosis between sexual partners — 0 = negative, 1 = positive (IgG levels confirm); 2-3 = unclear, excluded. Partner status removed (pattern rule); direct exposures: raw meat, garden soil, vegetables, cat breeding, number of cats.
- **poland_forensic_alcohol** (10.6084/m9.figshare.5226496): Latent Toxoplasma infection and excessive ethanol consumption (forensic autopsies)

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `07_figshare.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/07_figshare/07_figshare.ipynb`

Folder layout: `07_figshare.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

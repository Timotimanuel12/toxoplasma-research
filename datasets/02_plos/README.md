# 02_plos: PLOS journals (PLOS ONE, PLOS Neglected Tropical Diseases, PLOS Global Public Health)

Participant-level supplementary files (PLOS requires underlying data). Found through the Europe PMC search below and downloaded from the PMC Open Access S3 bucket.

**How the journal datasets were found.** Europe PMC was searched for every open-access paper with supplementary files whose title/abstract mentions Toxoplasma, toxoplasmosis, gondii, TORCH, multiplex serology or pathogen burden (3,482 papers). Their data files (xlsx, xls, csv, sav, dta, ...) were downloaded from the PMC Open Access S3 bucket (1,407 papers had data files, 6.6 GB) and every table with >= 80 rows was scanned for a Toxoplasma result column (607 tables, 285 papers). Human, participant-level tables with at least two IgG classes were then checked by hand. Scripts: `datasets/_harvest/`.

## Results (ROC-AUC, 5-fold x 3 CV)

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| indonesia_pregnant | Indonesia | Pregnant women | 182 | 60 | 17 | 0.865 | 0.865 | 0.863 | **0.865** |
| cameroon_hiv | Cameroon | People living with HIV | 325 | 164 | 20 | 0.641 | 0.623 | 0.602 | **0.641** |
| myanmar_slaughterhouse | Myanmar | Slaughterhouse workers | 139 | 78 | 46 | 0.682 | 0.662 | 0.618 | **0.682** |
| philippines_cebu | Philippines | General population | 924 | 244 | 16 | 0.672 | 0.688 | 0.671 | **0.688** |
| ghana_sickle_cell | Ghana | Sickle cell disease patients | 156 | 83 | 10 | 0.721 | 0.700 | 0.637 | **0.721** |
| usa_baltimore_psychiatric | USA | Schizophrenia, bipolar and control adults; plasma antibody panel | 2660 | 544 | 9 | 0.709 | 0.691 | 0.676 | **0.709** |
| mexico_insomnia | Mexico | Adults with insomnia and matched controls | 1154 | 117 | 8 | 0.736 | 0.725 | 0.702 | **0.736** |
| cambodia_women_multiplex | Cambodia | Women 15-39 y, national population-based serosurvey | 2150 | 128 | 25 | 0.518 | 0.571 | 0.544 | **0.571** |
| brazil_pregnant_vertical | Brazil | Pregnant women in primary care | 473 | 322 | 9 | 0.477 | 0.527 | 0.543 | **0.543** |
| kenya_hospital_lab | Kenya | Hospital laboratory requests | 1376 | 444 | 2 | 0.559 | 0.547 | 0.548 | **0.559** |
| france_sleep_apnea | France | Adults with obstructive sleep apnea | 82 | 55 | 41 | 0.561 | 0.647 | 0.592 | **0.647** |
| brazil_hiv_urine | Brazil | People living with HIV | 199 | 173 | 4 | 0.654 | 0.673 | 0.639 | **0.673** |
| brazil_leprosy_enl | Brazil | Leprosy patients | 103 | 76 | 0 |  |  |  |  |
| usa_women_financial_risk | USA | Women tested for Toxoplasma; lottery-choice task | 79 | 39 | 3 | 0.579 | 0.567 | 0.590 | **0.590** |

## Datasets

- **indonesia_pregnant** (10.1371/journal.pone.0245572): Seroprevalence and risk factors of T. gondii in pregnant women, Makassar
- **cameroon_hiv** (10.1371/journal.pone.0256947): Risk factors and CD4 count in people living with HIV, Bamenda
- **myanmar_slaughterhouse** (10.1371/journal.pone.0284352): Seroprevalence and risk factors among slaughterhouse workers, Yangon
- **philippines_cebu** (10.1371/journal.pone.0217989): Seroprevalence and risk factors, general population of Cebu
- **ghana_sickle_cell** (10.1371/journal.pone.0346155): T. gondii seropositivity in sickle cell disease patients, Ho
- **usa_baltimore_psychiatric** (10.1371/journal.pone.0116696): Infection and inflammation in schizophrenia and bipolar disorder (Johns Hopkins) — Label = Toxoplasma IgG z-score > 1.75; other-pathogen antibodies (CMV, HSV-1, HHV-6) as biomarkers.
- **mexico_insomnia** (10.1371/journal.pone.0266214): Toxoplasma gondii infection and insomnia: case-control seroprevalence study, Durango
- **cambodia_women_multiplex** (10.1371/journal.pntd.0004699): Multiplex bead assays for parasitic diseases in a national serosurvey of women — Toxoplasma SAG2A IgG on a multiplex bead assay; inputs = region, urban, age group + antibodies to Strongyloides, malaria, cysticercosis and lymphatic filariasis.
- **brazil_pregnant_vertical** (10.1371/journal.pone.0141700): Toxoplasmosis in pregnant women and vertical transmission, Brazil — Every maternal / newborn laboratory result removed; inputs are age, residence, antenatal care, pregnancies, abortions.
- **kenya_hospital_lab** (10.1371/journal.pone.0204867): Seroprevalence data at a private teaching hospital, Nairobi (Toxoplasma sheet) — Only age, sex and patient type (in/outpatient) recorded.
- **france_sleep_apnea** (10.1371/journal.pone.0235463): Chronic toxoplasmosis and sleepiness in obstructive sleep apnea
- **brazil_hiv_urine** (10.1371/journal.pntd.0009199): Detection of toxoplasmic encephalitis in HIV patients with urine nanoparticles — Urine antigen tests removed (they detect the parasite).
- **brazil_leprosy_enl** (10.1371/journal.pone.0300704): Toxoplasma infection and IgE in erythema nodosum leprosum — Single input (leprosy reaction type).
- **usa_women_financial_risk** (10.1371/journal.pone.0136716): No association between Toxoplasma infection and financial risk taking in females — Trial-level file collapsed to one row per woman.

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

## Run

Open `02_plos.ipynb` (in this folder) in Jupyter / VS Code and run all cells, or from the repository root:
`jupyter nbconvert --to notebook --execute --inplace datasets/02_plos/02_plos.ipynb`

Folder layout: `02_plos.ipynb` (one section per dataset: where it comes from, how the label is coded, what is removed; then the evaluation and a chart), `raw/` (original files), `processed/<dataset>.csv` (model-ready inputs + `target_toxo_pos`), `results.csv`.
Cite the original paper / repository DOI when using a dataset.

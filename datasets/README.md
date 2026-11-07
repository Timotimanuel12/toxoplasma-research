# Toxoplasma datasets by source

44 human, participant-level Toxoplasma IgG datasets from 10 sources (69,716 people, 15,502 seropositive); 40 have enough cases to evaluate. One folder per source; every folder uses the same evaluation protocol, so ROC-AUCs are comparable. (People counts for NHANES overlap: the 2009-2014 pooled cohort contains the 2013-2014 adults; without the overlap the total is 64,710 people.)

## Sources

| folder | source | datasets | evaluated | people | positives | best ROC-AUC |
|---|---|---|---|---|---|---|
| `01_nhanes` | CDC NHANES (U.S. National Health and Nutrition Examination Survey) | 4 | 4 | 43,664 | 8,743 | 0.742 |
| `02_plos` | PLOS journals (PLOS ONE, PLOS Neglected Tropical Diseases, PLOS Global Public Health) | 14 | 13 | 10,002 | 2,527 | 0.865 |
| `03_bmc_springer_nature` | BMC / Springer Nature journals (BMC Infectious Diseases, Parasites & Vectors, ...) | 3 | 2 | 1,360 | 228 | 0.632 |
| `04_peerj` | PeerJ | 2 | 2 | 1,598 | 148 | 0.892 |
| `05_mendeley_data` | Mendeley Data (Elsevier data repository) | 4 | 3 | 1,119 | 247 | 0.670 |
| `06_other_journals` | Other journals (ASTMH, Frontiers, Elsevier Heliyon, Wiley/Hindawi) | 4 | 4 | 680 | 303 | 0.702 |
| `07_figshare` | Figshare | 7 | 7 | 5,697 | 1,458 | 0.659 |
| `08_zenodo` | Zenodo (CERN) | 3 | 3 | 3,806 | 1,355 | 0.784 |
| `09_dryad` | Dryad | 2 | 1 | 1,490 | 343 | 0.571 |
| `10_harvard_dataverse` | Harvard Dataverse | 1 | 1 | 300 | 150 | 0.672 |

## Every evaluated dataset, best first

| dataset | country | setting | n | pos | features | LogReg | RF | XGBoost | best |
|---|---|---|---|---|---|---|---|---|---|
| 04/china_gynaecological_tumour | China | Women with gynaecological tumours + matched controls | 920 | 69 | 12 | 0.839 | 0.892 | 0.877 | **0.892** |
| 02/indonesia_pregnant | Indonesia | Pregnant women | 182 | 60 | 17 | 0.865 | 0.865 | 0.863 | **0.865** |
| 08/medical_students_latam | Latin America (Spanish-language) | Medical students; questionnaire on cats, meat, hand-washing, water | 640 | 221 | 16 | 0.767 | 0.784 | 0.780 | **0.784** |
| 08/france_blood_donors | France | Blood donors | 1995 | 1008 | 15 | 0.747 | 0.734 | 0.740 | **0.747** |
| 01/nhanes_2009_2014_pooled | USA | Adults 18+, 2009-2014 pooled, same assay; baseline + markers | 14649 | 2495 | 58 | 0.726 | 0.738 | 0.742 | **0.742** |
| 02/mexico_insomnia | Mexico | Adults with insomnia and matched controls | 1154 | 117 | 8 | 0.736 | 0.725 | 0.702 | **0.736** |
| 02/ghana_sickle_cell | Ghana | Sickle cell disease patients | 156 | 83 | 10 | 0.721 | 0.700 | 0.637 | **0.721** |
| 01/nhanes_1999_2004 | USA | Adults 18-49, 1999-2004, CDC EIA IgG >= 7 IU/mL; demographics, housing, insurance, smoking | 8485 | 1235 | 15 | 0.720 | 0.715 | 0.718 | **0.720** |
| 01/nhanes_2013_2014_markers | USA | Adults 18+, 2013-2014, Bio-Rad IgG >= 33 IU/mL; baseline + lifestyle + 44 markers (the paper) | 5006 | 750 | 54 | 0.699 | 0.718 | 0.713 | **0.718** |
| 02/usa_baltimore_psychiatric | USA | Schizophrenia, bipolar and control adults; plasma antibody panel | 2660 | 544 | 9 | 0.709 | 0.691 | 0.676 | **0.709** |
| 01/nhanes_iii_1988_1994 | USA | Adults 18+, 1988-1994; includes cat, garden, meat-frequency, region and water questions | 15524 | 4263 | 37 | 0.691 | 0.706 | 0.703 | **0.706** |
| 06/iran_psychiatric_inpatients | Iran | Psychiatric inpatients + healthy controls | 200 | 86 | 43 | 0.683 | 0.702 | 0.702 | **0.702** |
| 02/philippines_cebu | Philippines | General population | 924 | 244 | 16 | 0.672 | 0.688 | 0.671 | **0.688** |
| 02/myanmar_slaughterhouse | Myanmar | Slaughterhouse workers | 139 | 78 | 46 | 0.682 | 0.662 | 0.618 | **0.682** |
| 02/brazil_hiv_urine | Brazil | People living with HIV | 199 | 173 | 4 | 0.654 | 0.673 | 0.639 | **0.673** |
| 10/ghana_ashanti_pregnant | Ghana | Pregnant women attending antenatal clinics | 300 | 150 | 18 | 0.672 | 0.650 | 0.608 | **0.672** |
| 05/mexico_testosterone | Mexico | Adults, serologically tested | 213 | 35 | 15 | 0.647 | 0.652 | 0.670 | **0.670** |
| 07/czech_slovak_cat_survey | Czechia/Slovakia | Online survey, adults with a past Toxoplasma test | 1371 | 368 | 16 | 0.636 | 0.659 | 0.654 | **0.659** |
| 07/czech_health_stress | Czechia | Online survey | 747 | 153 | 8 | 0.652 | 0.612 | 0.601 | **0.652** |
| 06/colombia_quindio_ocular | Colombia | Community survey with fundoscopy | 161 | 112 | 17 | 0.651 | 0.621 | 0.590 | **0.651** |
| 02/france_sleep_apnea | France | Adults with obstructive sleep apnea | 82 | 55 | 41 | 0.561 | 0.647 | 0.592 | **0.647** |
| 02/cameroon_hiv | Cameroon | People living with HIV | 325 | 164 | 20 | 0.641 | 0.623 | 0.602 | **0.641** |
| 07/czech_personality_borrelia | Czechia | Online survey (Facebook "Guinea pigs" community) | 962 | 210 | 26 | 0.634 | 0.607 | 0.616 | **0.634** |
| 03/iran_suicide_attempters | Iran | Suicide attempters and controls | 500 | 153 | 12 | 0.632 | 0.631 | 0.625 | **0.632** |
| 05/thailand_bangphae_children | Thailand | Healthy children and young adults, paired sera 2012 and 2015 | 573 | 24 | 5 | 0.615 | 0.580 | 0.585 | **0.615** |
| 08/rennes_congenital | France | Infants born to mothers with toxoplasmosis in pregnancy | 1171 | 126 | 1 | 0.550 | 0.601 | 0.610 | **0.610** |
| 07/poland_forensic_alcohol | Poland | Forensic autopsy cases | 535 | 248 | 10 | 0.547 | 0.593 | 0.548 | **0.593** |
| 02/usa_women_financial_risk | USA | Women tested for Toxoplasma; lottery-choice task | 79 | 39 | 3 | 0.579 | 0.567 | 0.590 | **0.590** |
| 04/china_leukemia_children | China | Children with leukaemia + matched controls | 678 | 79 | 11 | 0.544 | 0.584 | 0.588 | **0.588** |
| 07/czech_students_toxo_cmv_iq | Czechia | Healthy university students, serologically tested | 556 | 97 | 29 | 0.560 | 0.533 | 0.587 | **0.587** |
| 06/thai_myanmar_border_pregnant | Thailand/Myanmar | Refugee and migrant pregnant women | 199 | 62 | 18 | 0.506 | 0.584 | 0.532 | **0.584** |
| 07/czech_sexual_partners | Czechia | Couples tested together; exposure questionnaire | 1433 | 357 | 9 | 0.579 | 0.561 | 0.553 | **0.579** |
| 02/cambodia_women_multiplex | Cambodia | Women 15-39 y, national population-based serosurvey | 2150 | 128 | 25 | 0.518 | 0.571 | 0.544 | **0.571** |
| 09/usa_students_risky_business | USA | University students, saliva IgG test | 1293 | 325 | 3 | 0.571 | 0.508 | 0.534 | **0.571** |
| 02/kenya_hospital_lab | Kenya | Hospital laboratory requests | 1376 | 444 | 2 | 0.559 | 0.547 | 0.548 | **0.559** |
| 05/nigeria_pregnant | Nigeria | Pregnant women at antenatal clinics | 250 | 171 | 43 | 0.552 | 0.558 | 0.533 | **0.558** |
| 02/brazil_pregnant_vertical | Brazil | Pregnant women in primary care | 473 | 322 | 9 | 0.477 | 0.527 | 0.543 | **0.543** |
| 03/colombia_thyroid | Colombia | Thyroid disease cases and controls | 704 | 59 | 5 | 0.538 | 0.484 | 0.483 | **0.538** |
| 06/brazil_homeless | Brazil | Homeless adults in shelters | 120 | 43 | 43 | 0.510 | 0.378 | 0.390 | **0.510** |
| 07/czech_toxo_cmv_longitudinal | Czechia | Adults re-tested after several years | 93 | 25 | 77 | 0.396 | 0.406 | 0.365 | **0.406** |

**Same protocol for every dataset** (`datasets/common/protocol.py`): label = Toxoplasma IgG positive; every column that measures Toxoplasma itself (IgM, avidity, titre, OD, PCR, other kits) and identifiers are removed; logistic regression, random forest and XGBoost; ROC-AUC from 5-fold x 3 repeated stratified cross-validation with preprocessing inside the folds. Datasets with fewer than 20 positives or 20 negatives are catalogued but not evaluated.

**How the journal datasets were found.** Europe PMC was searched for every open-access paper with supplementary files whose title/abstract mentions Toxoplasma, toxoplasmosis, gondii, TORCH, multiplex serology or pathogen burden (3,482 papers). Their data files (xlsx, xls, csv, sav, dta, ...) were downloaded from the PMC Open Access S3 bucket (1,407 papers had data files, 6.6 GB) and every table with >= 80 rows was scanned for a Toxoplasma result column (607 tables, 285 papers). Human, participant-level tables with at least two IgG classes were then checked by hand. Scripts: `datasets/_harvest/`.

**Reading the numbers.** These are within-dataset cross-validated scores. Small datasets (n < 300) have wide uncertainty (SD 0.05-0.10, see each results.csv); datasets whose inputs are personality or cognitive scores measure possible consequences of infection rather than exposures; case-control studies can score high because the case group itself differs in prevalence.

## Notebooks

- `00_summary.ipynb`: this table + charts across all sources
- `visualizations.ipynb`: all charts in one notebook (also saved as PNGs in `figures/`)
- `<folder>/<folder>.ipynb`: one per source (01 also builds NHANES 1999-2004)
- `common/protocol.ipynb`, `common/folder_tools.ipynb`: the shared evaluation protocol and dataset builder
- `_harvest/harvest.ipynb`: how the datasets were found (searches, downloads, table scan)

Run everything from the repository root:

```
for nb in datasets/[0-9]*_*/[0-9]*.ipynb datasets/00_summary.ipynb; do jupyter nbconvert --to notebook --execute --inplace $nb; done
```

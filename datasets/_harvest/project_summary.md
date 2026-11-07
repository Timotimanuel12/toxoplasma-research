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

## Top 12 evaluated datasets (best ROC-AUC, 5-fold x 3 CV)

| dataset | country | n | pos | best |
|---|---|---|---|---|
| 04/china_gynaecological_tumour | China | 920 | 69 | 0.892 |
| 02/indonesia_pregnant | Indonesia | 182 | 60 | 0.865 |
| 08/medical_students_latam | Latin America (Spanish-language) | 640 | 221 | 0.784 |
| 08/france_blood_donors | France | 1995 | 1008 | 0.747 |
| 01/nhanes_2009_2014_pooled | USA | 14649 | 2495 | 0.742 |
| 02/mexico_insomnia | Mexico | 1154 | 117 | 0.736 |
| 02/ghana_sickle_cell | Ghana | 156 | 83 | 0.721 |
| 01/nhanes_1999_2004 | USA | 8485 | 1235 | 0.720 |
| 01/nhanes_2013_2014_markers | USA | 5006 | 750 | 0.718 |
| 02/usa_baltimore_psychiatric | USA | 2660 | 544 | 0.709 |
| 01/nhanes_iii_1988_1994 | USA | 15524 | 4263 | 0.706 |
| 06/iran_psychiatric_inpatients | Iran | 200 | 86 | 0.702 |

## Takeaways (2026-11-01)

- Highest scores come from studies that ask direct exposure questions in high-exposure settings (Indonesia 0.865, Latin American medical students 0.784) or from case-control designs where the case group itself shifts prevalence (China tumour 0.892).
- Large general-population sets stay near 0.70-0.74: NHANES 1999-2004 0.720, 2013-14 0.718, NHANES III 0.706, Baltimore 0.709, France donors 0.747.
- Many questionnaire datasets are small (n < 300) and score 0.55-0.68 with wide uncertainty; Czech online surveys use self-reported past test results.
- MDPI, Dryad, OSF and Kaggle had little or no usable human serology (Kaggle: ocular fundus images only).
- Repo location: `datasets/` (one folder per source, shared protocol in `datasets/common/`, master table `datasets/ALL_RESULTS.csv`).

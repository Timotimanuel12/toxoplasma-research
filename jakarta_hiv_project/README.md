# Jakarta HIV study (questionnaire + Toxoplasma IgG)

Questionnaire answers and Toxoplasma IgG results of 187 people living with HIV in the Jakarta area (Jakarta, Depok,
Bogor), interviewed September 2022 to August 2023. This is the data behind Ichwan et al. (ICISS 2025) and is used by
the team's LLM part (RQ4). Used with the permission of the research team.

**Sensitive data** (HIV, sexual behaviour, drug use). Do not share outside the team or upload it anywhere public.

## Folders
| Path | What it is |
|---|---|
| `data/raw/Toxo sheets - Raw Data.csv` | Google Forms export (answer text). Has date of birth: personal data. |
| `data/raw/Toxo sheets - cleanedData.csv` | An earlier cleaned copy (answer codes, other row order). Ages are 2-3 years too old. |
| `data/processed/jakarta_hiv_clean.csv` | **Use this one.** 187 rows x 91 columns, ids J001-J187, no personal data. |
| `data/processed/jakarta_hiv_data_dictionary.csv` | Every column: role, meaning, values, number of blanks. |
| `docs/questionnaire_Rizky_Fajar_Meirawan.pdf` | The questionnaire, with the answer codes. |
| `scripts/clean_jakarta_hiv.py` | Builds the processed files from the raw export, with checks. Run from this folder. |

## Things to know before using the data
- Outcome: `igg_positive` (123 positive, 63 negative); 1 borderline person is flagged (`igg_borderline`); exclude them.
- Never use `igg_titer` as an input: it is the blood test itself.
- Age is at the interview date; 8 people have no age (their birth date was the interview date).
- Nine 0-4 questions are mixed scales where 0 means "does not apply" (for example condom use: 0 = no such sex,
  1 = always uses a condom, 4 = never). Use the `_applies` and `_risk` columns, not the original code.
- Blank follow-up answers (tattoo, drug use, cat questions) mean "does not apply".
- The questions on cat contact, cleaning cat faeces, CD4 and HIV diagnosis date are not in the export.

## Results so far (November 2026)
- `scripts/baseline_check.py` -> `outputs/baseline_check.txt`: with corrected ages and coding, all questions give
  CV ROC-AUC 0.47-0.51, age alone 0.58 (the best). Baselines an LLM must beat: ROC-AUC 0.58 (age), Brier 0.224
  (everyone gets 66%), positive-class F1 0.796 (everyone called positive). Only plausible single signal: among people
  who eat raw vegetables, washing them less often goes with being positive (p = 0.01, one of about 70 tests).
- `scripts/standard_ml.py` -> `outputs/standard_ml_results.txt`: six simple models (logistic regression, decision
  tree, random forest, kNN, naive Bayes, SVM) vs two dummy baselines, 5-fold CV x 20. No model beats the dummies:
  ROC-AUC 0.47-0.54, balanced accuracy 0.48-0.51; random forest and SVM reach F1 0.79 only by calling almost
  everyone positive (dummy "always positive" = 0.796). One 70/30 split gives ROC-AUC anywhere from about 0.37 to 0.61.
- `scripts/cross_dataset.py` -> `outputs/cross_dataset.txt`: Makassar vs Jakarta on 8 shared questions.
  Makassar CV 0.79 (0.87 with all its questions), Jakarta 0.51. Train Makassar -> test Jakarta 0.50 [0.41, 0.58];
  the reverse 0.38 [0.30, 0.47]. Undercooked/grilled poultry: OR 9.4 in Makassar vs 0.8 in Jakarta (differ p < 0.0001);
  un-boiled water 9.1 vs 1.4 (p = 0.009). Partly real population differences, partly different question wording.

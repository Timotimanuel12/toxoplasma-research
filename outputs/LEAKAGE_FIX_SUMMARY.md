# Leakage fix for toxo_max_score.ipynb, Part B (steps 0-5)

Run: 2026-10-30, `python scripts/max_score_fixed.py --mode both` (from `toxo-rich-fix/`).
Raw outputs: `outputs/max_score_fixed_results.txt` (printed log) and `outputs/max_score_fixed_results.json` (all numbers, per-fold CV scores, per-fold factor lists).
Environment: Python 3.12.10, scikit-learn 1.7.2, XGBoost 3.4.1, NumPy 2.4.2, pandas 2.3.2, SciPy 1.16.2 (Windows). Richard's versions were not recorded (Linux).

## The problem
Step 4 picks "biological/exposure factors independently linked to Toxoplasma" (p < 0.05 after age/sex/birthplace adjustment)
once, on all 13,647 training people, then cross-validates steps 4-5 on those same people. Every validation fold helped
choose the factors, so the CV scores and p-values of steps 4-5 are optimistic. The test set (1,002 people) is not affected.

Fix (`per_fold`): repeat the selection inside each of the 15 CV folds, using only that fold's training people.

## 1. Reproduction (mode `once`) - passed
| step | CV ours | CV notebook | TEST ours | TEST notebook |
|---|---|---|---|---|
| 0 earlier best (2013-14 only) | 0.7227 | 0.7227 | 0.7126 | 0.7126 |
| 1 + 2009-2014 training rows | 0.7267 | 0.7267 | 0.7121 | 0.7121 |
| 2 + all earlier features, age spline | 0.7256 | 0.7256 | 0.7132 | 0.7132 |
| 3 + age interactions | 0.7303 | 0.7303 | 0.7143 | 0.7143 |
| 4 + selected bio/exposure factors | 0.7346 | 0.7346 | 0.7333 | 0.7333 |
| 5 + XGBoost ensemble | 0.7398 | 0.7398 | 0.7289 | 0.7297 |

- Same 26 factors, same order. Same 1,002 test people. Steps 0-4 identical to 4 decimals.
- Only XGBoost-dependent numbers differ slightly (expected, different XGBoost version/OS):
  step 5 test 0.7289 vs 0.7297; step 5 p 0.0001 vs 0.0002; final threshold 0.126 vs 0.136
  (TP/FN/FP/TN 111/39/337/515 vs 106/44/303/549); bootstrap gain +0.016 (-0.001 to +0.034) vs +0.017 (+0.000 to +0.034).
- Earlier model (step 0, no XGBoost) matches exactly: threshold 0.142, 98/52/288/564.

## 2. Fixed results (mode `per_fold`)
| step | CV leaky | CV fixed | p leaky | p fixed | TEST (unchanged) |
|---|---|---|---|---|---|
| 0 | 0.7227 | 0.7227 | - | - | 0.7126 |
| 1 | 0.7267 | 0.7267 | 0.0852 | 0.0852 | 0.7121 |
| 2 | 0.7256 | 0.7256 | 0.3769 | 0.3769 | 0.7132 |
| 3 | 0.7303 | 0.7303 | 0.0312 | 0.0312 | 0.7143 |
| 4 | 0.7346 | **0.7327** | 0.0058 | **0.0233** | 0.7333 |
| 5 | 0.7398 | **0.7381** | 0.0001 | **0.0007** | 0.7289 |

- Optimism from the leakage: about -0.002 ROC-AUC for steps 4 and 5. Small.
- Best step by CV is still step 5 (final model unchanged).
- Step 5 vs step 0 is still significant after the fix (corrected t-test p = 0.0007).
- Final model at the fixed threshold (0.131): sensitivity 0.727, specificity 0.622, balanced accuracy 0.674, TP/FN/FP/TN 109/41/322/530.
- Test bootstrap gain (step 5 vs step 0) is unchanged by the fix: +0.016, 95% CI -0.001 to +0.034 (final better in 97% of resamples).
  Borderline: the CI touches or crosses zero in both runs.

## 3. Factor stability across the 15 folds
- Per fold: 21-25 factors (mean 22.7) vs 26 on all training people.
- Always picked (15/15), 17 factors: hav_natural_infection, hav_antibody, hsv1, hsv2, toxocara_mfi_log, cotinine_log,
  well_or_spring_water, blood_lead_log, no_health_insurance, food_insecurity, work_construction_soil, missing_teeth,
  not_us_citizen, crowding, home_rented, spanish_interview, household_size.
- Mostly picked: eosinophil_pct 14/15, wbc_count 13/15, no_tap_water 11/15.
- Unstable (picked in about half the folds or fewer): work_food_service 9, blood_mercury_log 8, lymphocyte_pct 6, inject_drugs_ever 6, hev_igg 5, hav_vaccinated 3.
- Never in the 26 but picked in some folds: work_agriculture 3, oysters_30d 3, hcv_antibody 1, shellfish_30d 1, poultry_g_log 1,
  organ_sausage_lunchmeat_g_log 1, beef_g_log 1.

## 4. Impact on the paper (main.pdf) - step 6 assessment
| Where in main.pdf | Paper says | After the fix |
|---|---|---|
| Abstract | CV ROC-AUC 0.740 (p < 0.001) | 0.738 (p < 0.001 still true: 0.0007) |
| Sec. IV-D | 0.740, corrected p = 0.0002 vs step 0 | 0.738, p = 0.0007 |
| Table IV step 4 | CV 0.735 | 0.733 |
| Table IV step 5 | CV 0.740 | 0.738 |
| Table IV test column | 0.713 ... 0.730 | unchanged |
| Conclusion | 0.740 in cross-validation | 0.738 |
| Sec. IV-D / IV-E threshold | 0.136; 106/44/303/549; sens 70.7%, spec 64.4%, BA 67.6%; 409 flagged (40.8%), PPV 25.9%, 7.4% (44/593) | changes slightly; must be recomputed in one environment (ours, XGBoost 3.4.1: 0.131; 109/41/322/530; sens 72.7%, spec 62.2%, BA 67.4%; 431 flagged (43.0%), PPV 25.3%, 7.2% (41/571)) |
| Bootstrap +0.017 [0.000, 0.034] | already says "lower bound touches zero" | unaffected by the fix (ours: +0.016 [-0.001, +0.034]) |
| Fig. 10a | step 4-5 CV bars | redraw with 0.733 / 0.738 |

Not affected: Table III (uses all 44 markers, no selection), train_harder / Optuna (all 44 markers, no selection),
the bio notebook "+selected" models (not reported in the paper), the main message ("the limit is information, not the learner").

Other wording issues (not leakage):
- Sec. III-D step (4) "markers that stayed associated in the training data" = 26 markers on pooled 2009-2014 training data,
  not the 17 of Fig. 5 (descriptive screen on all 5,006 adults of 2013-2014, test people included). The two lists overlap in 14:
  Fig. 5 has shellfish, blood mercury, eosinophils; the 26 / stable-17 have hepatitis A (natural, antibody), well/spring water, etc.
  The paper should say "26" and state that Fig. 5 is descriptive and not used for model building.
- Optuna CV 0.740 / 0.741 is the best of 60 trials on the same folds (tuning optimism); its test 0.719 / 0.724 shows this.
- "ROC-AUCs above 0.80 [15], [16]": [15] Polanunu 2021 reports odds ratios, not ROC-AUC. Our own analysis of its public data
  (makassar_project) gives CV ROC-AUC 0.87 (LR, all features) - that can support the sentence if cited as our analysis. [16] Wang 2022: check.
- NHANES III (cats, gardening, meat; CV ~0.70, +0.003 to +0.011) is computed in the notebook but not reported; it supports the
  "information" message (general current exposure questions do not help; specific ones in high-exposure settings do).

## Not covered by this run
- Interaction medians still taken over the whole pooled file (minor issue #4, kept for exact reproduction).
- Bio notebook "+selected" models (same leakage pattern) not yet fixed.
- Issues #2 (26 vs 17 markers in the paper) and #3 (NHANES III omitted, ">0.80" sentence) are wording issues, not code.

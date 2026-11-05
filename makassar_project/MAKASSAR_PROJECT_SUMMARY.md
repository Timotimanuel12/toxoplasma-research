# Makassar Toxoplasma Project: Summary

**Status (26 Oct 2026):** data cleaned and verified; baselines, controls and self-distillation finished.
**Main finding:** self-distillation makes the over-confident MLP better calibrated and clearly beats label smoothing, but it adds nothing beyond ordinary L2 regularisation, and it does not change logistic regression. Logistic regression remains the best model (ROC-AUC 0.874).

---

## 1. Data

- **Source:** Polanunu, Wahyuni & Hamid (2021), *PLoS ONE* 16(6): e0245572, supporting file S1 (SPSS), CC BY 4.0. Cite the paper when using the data.
- **Raw file:** `data/raw/polanunu2021_S1.sav`: 184 pregnant women from 9 primary health centres (Puskesmas) in Makassar, Sept–Oct 2020.
- **Clean file:** `data/processed/makassar_clean.csv` (23 columns) + `data/processed/makassar_data_dictionary.csv` (role and coding of every column).
- **Target:** IgG positive (index > 1.1): 60 of 184 (32.6%). The 2 borderline women (index 0.9–1.1) are flagged (`igg_borderline`) and **excluded from modelling, like the paper**, giving 182 women, 60 positive.
- **Not features:** `igg_index` (same blood test, leaks the target), IgM (nobody positive), `id` / `site` / `site_name` (identification; `site` is used for the health-centre check).

**Verified against the paper** (`scripts/verify_against_paper.py`): all IgG/IgM results, age, education, all 9 site sizes and 14 of 15 rows of Table 3 match exactly.
- Table 1 of the paper contradicts its own Table 3 for employed (32 vs 37), primigravida and miscarriage; the data matches Table 3.
- One positive woman sits at a different site (Antang/Tamalate).
- The adjusted odds ratios **could not be reproduced exactly**: ours are cat contact 8.0, satay 7.5, unboiled water 5.2, vs the paper's 10.5 / 9.7 / 6.0. The same three factors are significant.
- Household size is in the paper but not in the public file.

---

## 2. Feature sets

| Set | Features |
|---|---|
| **A: background (7)** | age, trimester, gravidity, parity, education, abortion history, employed |
| **B: exposures (10)** | cat contact, raises cat, cleans litter, soil contact, ate chicken satay (undercooked meat), ate raw vegetables, drank un-boiled water, drinks filtered gallon water, washing-water source, cooking-water source |
| **C: all (17)** | A + B, the main set for self-distillation |

---

## 3. Scripts (run from inside `makassar_project/`, in this order)

| Step | Script | Output |
|---|---|---|
| 1 | `clean_makassar.py` | clean CSV + data dictionary |
| 1b | `verify_against_paper.py data/raw/polanunu2021_S1.sav data/processed/makassar_clean.csv` | printed check |
| 2 | `train_baselines.py` | `outputs/baseline_results.txt`, `baseline_fold_scores.csv` |
| 3 | `train_controls.py` | `outputs/controls_results.txt`, `controls_fold_scores.csv` |
| 4 | `train_self_distillation.py` (~35 min) | `outputs/self_distillation_results.txt`, `self_distillation_fold_scores.csv` |

`common.py` holds the shared setup:
- data loading and feature sets
- preprocessing (scaling + one-hot), the 100 CV folds and the corrected t-test
- **`TorchMLP`**: a PyTorch MLP that accepts soft labels (1 hidden layer of 16 units, weight decay 0.001, learning rate 0.01, 300 full-batch epochs)
- **`SoftLabelLogReg`**: logistic regression that accepts soft labels via the duplicate-row trick; identical to normal LR on hard labels (checked: max difference 7e-15)
- **`evaluate()`**: the CV engine; training targets can change inside each fold, but scoring always uses the true labels

Requires pandas, scikit-learn 1.7, xgboost, torch (CPU) and pyreadstat.

---

## 4. Evaluation design
- **Stratified 5-fold CV repeated 20 times = 100 folds**, identical for every model and step (seed 42). No separate test set, because 182 women is too few.
- **Corrected t-test** (Nadeau–Bengio).
  - ROC-AUC: "meaningful" = gain of at least 0.01 **and** p < 0.05; "worse" = lower and p < 0.05.
  - Brier: "better" / "worse" = p < 0.05.
- Metrics: ROC-AUC, PR-AUC, Brier score, calibration slope (1 = ideal, below 1 = over-confident).
- **Site check:** leave-one-site-out (train on 8 health centres, test on the 9th), reported as pooled ROC-AUC.
- All model settings fixed in advance (no tuning). Every condition is reported (no picking the best).

---

## 5. Results

### Step 2: baselines (ROC-AUC, mean of 100 folds)

| Model | A background | B exposures | C all | Brier (C) |
|---|---|---|---|---|
| Dummy | 0.500 | – | – | 0.221 |
| **Logistic regression** | 0.497 | 0.866 | **0.874** | **0.143** |
| Random forest | 0.497 | 0.867 | 0.858 | 0.152 |
| XGBoost | 0.493 | 0.853 | 0.864 | 0.154 |
| MLP | 0.482 | 0.820 | 0.842 | 0.192 |

- Background predicts nothing (~0.50). Exposures add +0.34 to +0.37 (meaningful, p < 0.001). Adding background to exposures does not help.
- No model beats logistic regression.
- Leave-one-site-out pooled ROC-AUC 0.84–0.87; single sites range 0.67–1.00 (5–37 women each).

### Step 3: controls (feature set C)

| Student | Condition | ROC-AUC | Brier | Slope |
|---|---|---|---|---|
| MLP | hard labels | 0.842 | 0.192 | 0.22 |
| MLP | label smoothing ε = 0.05 / 0.10 / 0.20 | 0.790 / 0.774 / 0.774 (worse) | 0.184 / 0.183 / 0.179 | 0.43 / 0.53 / 0.83 |
| MLP | weight decay 0.01 / **0.1** | 0.850 / **0.869** | 0.165 (better) / **0.148** | 0.45 / 1.69 |
| LR | hard labels | 0.874 | 0.143 | 0.74 |
| LR | label smoothing ε = 0.05 / 0.10 / 0.20 | 0.874 / 0.874 / 0.874 | 0.138 / 0.138 / 0.150 | 1.03 / 1.33 / 2.08 |
| LR | C = 0.1 / 0.01 | 0.875 / 0.854 | 0.140 / 0.175 (worse) | 1.34 / 3.20 |

**Step 4a check (one-off):** the label-smoothing drop for the MLP stays under other training settings, so it is not a training artefact:

| Learning rate | Epochs | Hard labels ROC-AUC | Label smoothing ε = 0.1 ROC-AUC | Difference |
|---|---|---|---|---|
| 0.01 | 300 | 0.842 | 0.774 | −0.068 |
| 0.01 | 1,000 | 0.840 | 0.775 | −0.065 |
| 0.001 | 1,000 | 0.850 | 0.808 | −0.042 |
| 0.001 | 3,000 | 0.842 | 0.785 | −0.057 |

Likely reason: many women share identical answers, and smoothing caps every confident pattern at 0.9, which erases differences and creates ties. The MLP settings were kept unchanged.

### Step 4: self-distillation (teacher = same model type, out-of-fold soft labels)

Student target = α × true label + (1 − α) × teacher probability (softened by temperature T). Grid: α ∈ {0.3, 0.5, 0.7} × T ∈ {1, 2, 4}, plus generations 2–3 at α = 0.5, T = 2.

| | ROC-AUC | Brier | Slope |
|---|---|---|---|
| MLP hard labels | 0.842 | 0.192 | 0.22 |
| **MLP self-distillation (9 settings)** | 0.839–0.862 | 0.148–0.170 | 0.36–0.92 |
| MLP generation 2 / 3 | 0.847 / 0.841 | 0.154 / 0.154 | 0.72 / 0.87 |
| MLP weight decay 0.1 (control) | 0.869 | 0.148 | 1.69 |
| LR self-distillation (11 conditions) | 0.874–0.877 | 0.137–0.149 | 0.81–2.09 |

1. **MLP vs hard labels:** Brier better by 0.022–0.044 (significant in 7 of 11 conditions); ROC-AUC +0.005 to +0.020 (not significant).
2. **MLP vs label smoothing:** ROC-AUC +0.07 to +0.09, **significant** (p = 0.002–0.018). Per-person soft labels avoid the ties that uniform smoothing creates.
3. **MLP vs weight decay 0.1:** never better (ROC-AUC −0.006 to −0.029, Brier +0.000 to +0.022, none significant). The best setting (α = 0.3, T = 4) ties on Brier (0.148) with better calibration (0.92 vs 1.69) but ranks lower (0.857 vs 0.869).
4. **Generations:** ranking drifts down, calibration keeps improving. Repeating mostly adds softening.
5. **Pattern:** a higher T means better calibration and slightly worse ranking; a lower α (more teacher) means a better Brier score.
6. **LR:** no effect (ROC-AUC ±0.003, Brier ±0.006, none significant). It was already well calibrated.

### Example women (out-of-fold predicted risk)

| Woman | Profile | True | MLP hard | MLP self-distilled (α = 0.5, T = 2) | MLP weight decay 0.1 | LR |
|---|---|---|---|---|---|---|
| KB010 | all exposures | 1 | 1.00 | 1.00 | 0.72 | 0.98 |
| KB001 | no exposures | 0 | 0.00 | 0.00 | 0.07 | 0.00 |
| KB030 | only satay, positive | 1 | 0.08 | 0.07 | 0.23 | 0.39 |
| KB006 | all exposures, negative | 0 | 1.00 | 0.99 | 0.72 | 0.73 |
| % extreme predictions (< 0.05 or > 0.95) | | | 74% | 50% | 0% | 35% |

Self-distillation reduces extreme predictions, but it keeps the teacher's confident mistakes (KB006 = 0.99).

---

## 6. Conclusion
Self-distillation behaves like regularisation, which fits the theory. On this small tabular dataset it helps an over-confident, untuned MLP and beats uniform label smoothing, but it gives **no benefit beyond plain L2 regularisation** and **none for logistic regression**. This is a clear, reportable null result.

**Caveats:**
- 182 women with 60 positives, so low power for ROC-AUC differences.
- 22 self-distillation comparisons without multiple-testing correction, although the direction is consistent across the grid.
- One city and one population (pregnant women attending public health centres), with self-reported exposures (recall bias possible).

---

## 7. Possible next steps
1. **Size experiment:** repeat on NHANES subsamples (184, 500, 1,000, 4,004) to see whether self-distillation helps more with more or less data. This fits Mr Feri's "multiple domains" plan.
2. **Self-distillation on top of weight decay:** does it add anything to a well-regularised MLP (3–4 conditions)?
3. Other methods from Mr Feri's list: knowledge distillation (e.g. LR teacher → MLP student), multi-teacher KD, deep mutual learning (the code in `common.py` / `train_self_distillation.py` supports other teachers).
4. **Write-up.** Verify these references before citing: Hinton et al. 2015 (knowledge distillation), Furlanello et al. 2018 (Born-Again Networks), Zhang et al. 2019 (Be Your Own Teacher), Mobahi et al. 2020 (self-distillation as regularisation), Müller, Kornblith & Hinton 2019 and Yuan et al. 2020 (link with label smoothing), plus Polanunu et al. 2021 (data).

## 8. Added November 2026 (for the LLM paper)
- **Standard ML report** (`scripts/standard_ml.py` -> `outputs/standard_ml_results.txt`), same format as the Jakarta HIV study:
  six simple models + two dummies, all 17 questions, 5-fold CV x 20. Logistic regression best: ROC-AUC 0.874,
  accuracy 77.6% (dummy "always negative" 67.0%), balanced accuracy 0.73. Confusion matrix (threshold 0.5): finds
  39 of 60 infected, 100 of 122 uninfected correct. One 70/30 split: ROC-AUC 0.80-0.92 over 100 seeds (always far
  above chance; Jakarta: 0.37-0.61).
- **LLM package** (`scripts/make_llm_handoff.py` -> `outputs/llm_handoff/`, zipped): 182 English text profiles,
  separate answer key, out-of-fold LR risk per woman (pooled ROC-AUC 0.878), README with fairness rules.
- **Cross-population test** with the Jakarta HIV study: see `../jakarta_hiv_project/outputs/cross_dataset.txt`
  (models do not transfer; undercooked poultry OR 9.4 here vs 0.8 in Jakarta).

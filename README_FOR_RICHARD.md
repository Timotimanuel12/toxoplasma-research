# Paper fixes for "ML Prediction of Toxoplasma gondii Seropositivity Using NHANES" (main.pdf)

Hi Richard. Timothy and I (with Claude) checked your `rich` branch against main.pdf. Your code and the paper agree:
every number we checked reproduces (same 1,002 test people, same 26 markers, CV 0.7227 to 0.7398, test 0.7126 to 0.7333).
We found one leakage problem and a few wording issues. They are all fixable and the main message
("the limit is information, not the learner") stays the same.

Nothing has been pushed or committed to GitHub. Everything is in this folder.

---

## 1. What we found

| # | Issue | Effect |
|---|---|---|
| 1 | **Leakage in step 4.** In `toxo_max_score.ipynb` (Part B, cell 10) the markers are selected once on all 13,647 training people, then steps 4-5 are cross-validated on those same people. Every validation fold helped choose the markers. (The notebook's last cell already admits this; the paper doesn't.) | CV of steps 4-5 is about 0.002 too high. Test set not affected. |
| 2 | **26 vs 17 markers.** Step 4 uses 26 markers (pooled 2009-14 training data). Readers will think it uses the 17 in Fig. 5, which come from a descriptive screen on all 5,006 adults (test people included). | Wording only. |
| 3 | **">0.80 [15], [16]" sentence.** [15] (Polanunu 2021, Makassar) reports odds ratios, not ROC-AUC. [16] (Wang 2022) not checked. Also, the NHANES III result (cats, gardening, meat: CV 0.691 to 0.702) is computed in the notebook but not in the paper. | Wording only. |
| 4 | Minor: the interaction medians (`age_x_edu`, `age_x_pir`) are computed on the whole pooled file, test people included. | Negligible; kept as is. |

Not affected: Table III, the Optuna / "training harder" results (they use all 44 markers, no selection), the bio-notebook
"+selected" models (not in the paper).

## 2. The fix and its results

`scripts/max_score_fixed.py` copies Part B of `toxo_max_score.ipynb` exactly and runs the step-4 marker selection two ways:
`once` (as in your notebook) and `per_fold` (selection repeated inside each of the 15 folds, using only that fold's training people).

| Step | CV (yours) | CV fixed | p (yours) | p fixed | Test (unchanged) |
|---|---|---|---|---|---|
| 4 + associated markers | 0.7346 | **0.7327** | 0.0058 | **0.0233** | 0.7333 |
| 5 + XGBoost ensemble | 0.7398 | **0.7381** | 0.0002 | **0.0007** | 0.7297 |

- Steps 0-3 do not change. Step 5 is still the best step and still significant.
- Per fold, 21-25 markers are selected; 17 are selected in all 15 folds; `hav_vaccinated` (3/15) and `hev_igg` (5/15) are unstable.
- Full details: `outputs/LEAKAGE_FIX_SUMMARY.md`. Raw log and all numbers: `outputs/max_score_fixed_results.txt` / `.json`.

**One thing only you can do:** we ran this on Windows with XGBoost 3.4.1. Steps 0-4 match your numbers exactly, but XGBoost
results differ very slightly (step 5 test 0.7289 vs your 0.7297; threshold 0.126 vs your 0.136). So the **threshold numbers**
(threshold, sensitivity, specificity, TP/FN/FP/TN, flagged, PPV, NPV) should be recomputed on **your** machine, so they
match the rest of the paper. They are marked `% TODO` in `paper/main.tex`.

## 3. What is already changed in `paper/main.tex`

Compare with `paper/main_original.tex` (the version Timothy got from Prism):

- CV 0.740 -> **0.738** (abstract, Sec. IV-D, conclusion); p = 0.0002 -> **0.0007**; Table IV steps 4/5 -> 0.733 / 0.738.
- Sec. III-D: step 4 = "the markers that stayed associated in the training data (26), with this selection repeated inside every cross-validation fold".
- Sec. IV-C: "This screen is descriptive and was not used to build any model."
- Optuna: "(best of 60 trials, so slightly optimistic)".
- ">0.80 [15], [16]" replaced by the NHANES III result (0.691 -> 0.702) and our Makassar result (logistic regression 0.87 with
  exposure questions vs 0.50 without; source: `makassar_project/outputs/baseline_results.txt` on the `makassar` branch of
  toxoplasma-research). `wang2022` is commented out; put it back only if it reports a ROC-AUC.
- **Page limit is 6 pages including references.** To fit, four figures were removed (their numbers stay in the text/tables):
  old Fig. 3 (models, = Table III), Fig. 4 (fold box plot), Fig. 9 (confusion matrices, had the old threshold) and
  Fig. 10 (improvement steps, = Table IV, had the old CV values). They are in `paper/unused_figures/`.
- The figure PDFs were cut out of your compiled main.pdf (the figure files are not in the repo). If you have the original
  figure files, use those instead (same names).

## 4. How to do the rest with Claude Code

1. Unzip this folder somewhere outside your repo, e.g. `~/handoff_for_richard/`.
2. In your repo, check out `rich` and make a new branch, e.g. `git checkout -b leakage-fix`.
3. Start Claude Code in the repo root and paste the prompt below (change the path to where you unzipped this folder).
4. Check what Claude changes before you commit. Please tell Timothy before you merge or push anything.

### Prompt to paste into Claude Code

```
I'm fixing our NHANES Toxoplasma paper. A teammate checked my code and left a handoff folder at
<PATH>/handoff_for_richard. Read <PATH>/handoff_for_richard/README_FOR_RICHARD.md and
<PATH>/handoff_for_richard/outputs/LEAKAGE_FIX_SUMMARY.md first, then do these steps one at a time.
Stop after each step, show me the result, and wait for my OK before the next one. Don't push anything.

1. Copy <PATH>/handoff_for_richard/scripts/max_score_fixed.py into scripts/. Read it next to
   research_r/toxo_max_score.ipynb (Part B, cells 10-14) and confirm it reproduces Part B exactly,
   except the per_fold marker selection.
2. Run: python scripts/max_score_fixed.py --mode both  (about 15-25 minutes). Check the "once" table
   matches the notebook (26 markers, CV 0.7346 / 0.7398, test 0.7333 / 0.7297, threshold 0.136,
   TP/FN/FP/TN 106/44/303/549). Then show me the per_fold results next to the numbers in the handoff summary.
3. From the per_fold "final model" row, compute: threshold, sensitivity, specificity, balanced accuracy,
   TP/FN/FP/TN, number flagged (TP+FP) and % of 1,002, PPV = TP/(TP+FP), and the chance of infection after a
   negative result = FN/(FN+TN). Compare with the earlier model (98/52/288/564).
4. Copy <PATH>/handoff_for_richard/paper/ into paper/ (or wherever I keep the LaTeX). Fill in every
   "% TODO (leakage fix)" in main.tex with the numbers from step 3 (Sec. IV-D threshold paragraph and
   Sec. IV-E), then delete those TODO comments. Keep the wording otherwise unchanged.
5. Update research_r/toxo_max_score.ipynb so step 4's selection runs inside each CV fold (like
   max_score_fixed.py), and update its last markdown cell ("What to write in the paper") to the fixed numbers.
6. Check reference wang2022 (PeerJ 10:e14569, 2022): does it report a ROC-AUC above 0.80? If yes, tell me
   and suggest where to cite it; if no, leave it commented out.
7. Show me a short summary of every change, so I can compile in Prism and check it is still 6 pages.
```

## 5. Files in this folder

```
README_FOR_RICHARD.md                 this file
scripts/max_score_fixed.py            the fix (run from the repo root; needs scripts/build_bio_features.py and data/processed/)
outputs/LEAKAGE_FIX_SUMMARY.md        all results, marker stability, impact on the paper
outputs/max_score_fixed_results.txt   printed log of our run
outputs/max_score_fixed_results.json  every number (per-fold CV, per-fold marker lists, thresholds, bootstrap)
paper/main.tex                        edited paper (6 figures + flow chart)
paper/main_original.tex               the version before our edits, for comparison
paper/fig_*.pdf                       the 6 figures used
paper/unused_figures/                 the 4 removed figures
```

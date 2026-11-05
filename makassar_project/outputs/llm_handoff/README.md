# Makassar data for the LLM screening test

A second test set for the LLM, next to the Ichwan data. It has 182 pregnant women from 9 health centres in Makassar,
Indonesia, and 60 of them (33%) are Toxoplasma IgG positive. The questionnaire asks directly about cats, undercooked
meat, soil and water, so it suits the LLM. We already have an ML result on these same women to compare with.

**Source:** N. F. A. Polanunu, S. Wahyuni, and F. Hamid, "Seroprevalence and associated risk factors of Toxoplasma
gondii infection among pregnant mother in Makassar, Indonesia," *PLoS ONE*, vol. 16, no. 6, e0245572, 2021.
The data is public (CC BY 4.0), so you can use it if you cite the paper. Two borderline blood results are left out,
as in the original paper.

## Files
| File | What it is |
|---|---|
| `profiles.csv` | `id`, `site`, `profile`: one English text profile per woman (83-103 words). Answers only, **no blood test result**. |
| `makassar_llm_text.csv` | **Use this one for the LLM run.** Same format and section style as your Jakarta input `toxo_llm_text.csv`: columns `row_id`, `text`, `serostatus`. `text` goes into the prompt's `{profile}`; **never put `serostatus` in the prompt** (it is the answer key). `row_id` = row of `makassar_clean.csv`; rows 28 and 149 (the 2 borderline women) are skipped, like row 129 in your Jakarta file. |
| `row_id_to_woman_id.csv` | Which woman each `row_id` is (e.g. row 2 = KB003). |
| `labels.csv` | `id`, `igg_positive` (1 = positive). **The answer key. Don't open it until the LLM has scored all 182 profiles.** |
| `ml_predictions.csv` | `id`, `lr_all`, `lr_exposures`: risk (0-1) from our logistic regression for each woman. Each score comes from a model that never saw her (5-fold CV repeated 20 times, averaged). `lr_all` uses all questions, `lr_exposures` uses only the exposure questions. |

## How the profiles are written
- Every profile uses the same template and mentions **every question in the same order, including the "no" answers**.
  No profile stands out just because of its wording.
- The words "Toxoplasma", "IgG", "positive" and "negative" never appear in a profile (the script checks this).
- "Filtered refill-gallon water" also counts as "un-boiled water", as in the original paper, so both often appear together.
- Example (KB003): *"A 20-year-old pregnant woman from Makassar, Indonesia, in her first trimester. This is her first
  pregnancy; she has no previous births, and she has no history of miscarriage. She finished senior high school and is
  not employed. Her household uses piped (tap) water for washing and cooking, and she drinks boiled water. She raises a
  cat, has contact with cats and cleans cat litter. She has no contact with soil. During pregnancy she ate undercooked
  chicken satay, ate raw vegetables and did not drink un-boiled water."*

## What to compare against
Our logistic regression on the same 182 women (scores from `ml_predictions.csv`):

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| LR, all questions (`lr_all`) | 0.878 | 0.787 | 0.138 |
| LR, exposure questions only (`lr_exposures`) | 0.861 | 0.782 | 0.138 |
| Guessing (everyone gets the same risk) | 0.500 | 0.330 | 0.221 |

Please score the LLM the same way: ROC-AUC, PR-AUC, and Brier score if it gives a risk between 0 and 1.
Because both models score the same women, you can also compare them person by person.

## Two rules to keep the test fair
1. Fix the prompt (and the paper library) **before** running on this data, and don't change it after seeing results.
2. Keep the Polanunu 2021 paper itself **out of the paper library**. It describes the results of these same women.
   (The same applies to the Ichwan paper for the Ichwan data.)

Made with `makassar_project/scripts/make_llm_handoff.py` (it can be rerun from the clean data).

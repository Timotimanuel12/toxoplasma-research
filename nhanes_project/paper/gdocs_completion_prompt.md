You are editing my research proposal in Google Docs, titled "Predicting Latent Toxoplasmosis: A Machine Learning Approach Integrating Demographic, Behavioral, and Health Questionnaires" (authors: Vickelsteins August Santoso and Timothy Jonathan Imannuel). Please complete and update it so that it matches our team's paper draft, which I have attached as main.pdf ("Machine Learning Prediction of Toxoplasma gondii Seropositivity Using NHANES Survey and Biological Marker Data"). If the PDF is not attached, use the fact sheet in Part 3 of this prompt, which contains everything you need.

# Part 1. How to work

1. Read the whole Google Doc first, then the attached paper.
2. Make every edit in **Suggesting mode**, so I can accept or reject each change.
3. Work section by section in the order of Part 4. Keep my section headings and structure, and add only the new section listed there (Results Summary).
4. When something needs my decision, or a fact is missing, add a **comment** instead of guessing.
5. When you finish, add one comment at the top of the document that lists what you changed and what still needs my decision.

# Part 2. Writing rules

- Use simple, clear English, in the same tone as my existing text. Short sentences, no hype.
- Make only the changes needed to match the paper. Keep my wording wherever it is still correct.
- Use numbers only from the paper or from Part 3. Never invent a number, a citation or a source.
- Citations use IEEE style: numbers in square brackets, numbered in order of first appearance in the text. Every number cited in the text must appear in the reference list, and every reference must be cited.
- Do not mix numbers from different analyses. All results must come from the paper (Part 3).
- Report the paper's results honestly:
  - Base the claims on **cross-validation**. Test-set differences smaller than about 0.045 are within noise (the paper says so itself). So do not present the test-set gains from the markers (+0.015 to +0.020) as real improvements.
  - Present the **biological markers as a way to understand what limits prediction**, not as a screening tool. Most markers need a blood test, which goes against the aim of avoiding blood tests. The practical model is the survey-based one.
  - When describing the "associated markers" step, say the markers were chosen **using the training data only**.
  - **Do not** copy the paper's sentence "Cohorts that asked these questions directly ... have reached ROC-AUCs above 0.80 [15], [16]". Those two references do not report ROC-AUCs. If you mention direct exposure questions, use this sentence instead: "Studies that asked directly about cats, undercooked meat and untreated water found strong associations with infection; for example, in public data from 184 pregnant women in Makassar, Indonesia [Polanunu et al. 2021], cat contact, undercooked chicken satay and un-boiled water were independently associated with seropositivity."
- Remove all content about large language models (LLMs). It is out of scope.

# Part 3. Fact sheet from the paper

**Data and cohort**
- Main cohort: NHANES 2013–2014, adults aged 18 or older with a Toxoplasma IgG result: 5,006 adults, 750 (15.0%) positive. IgG of 33 IU/mL or higher = positive; equivocal results (27–33 IU/mL) were coded negative, following CDC documentation.
- Stratified 80/20 split with a fixed seed: 4,004 training adults and 1,002 test adults (150 positive). The test adults are used only after each model is fixed.
- For the improvement steps, the 2009–2010 and 2011–2012 cycles were added to the training data (same assay and cut-off): 14,649 adults in total (17.0% positive), 13,647 training adults. The test set did not change.
- Seroprevalence fell across cycles: 19.3% (2009–2010), 16.8% (2011–2012), 15.0% (2013–2014).

**Feature sets**
- Baseline: age, sex, race/ethnicity, years in the U.S., education, poverty–income ratio, Toxocara serostatus.
- Lifestyle: self-rated health, sleep, smoking, alcohol, PHQ-9 depression score.
- Markers: 44 biological and exposure variables from 20 NHANES files:
  - other infections: hepatitis A (natural infection and vaccination separated), hepatitis E, B and C; HSV-1/2; HIV; Toxocara IgG level
  - blood markers: white cells, eosinophils, lymphocytes, monocytes, haemoglobin
  - environmental markers: blood lead, cadmium and mercury; cotinine
  - water and diet: well water, shellfish, fish, grams of each meat type in a 24-hour recall
  - housing, work and poverty: crowding, renting, interview language, citizenship, farm/food-service/construction work, food insecurity, insurance, drug use, missing teeth
- Toxoplasma IgM and avidity were excluded because they measure the infection itself.
- Each marker was also tested with logistic regression adjusted for age, sex and birthplace.

**Models and evaluation**
- Four learners with fixed settings:
  - logistic regression (C = 1)
  - random forest (500 trees, minimum leaf 10)
  - gradient boosting (300 iterations, depth 3, learning rate 0.05)
  - XGBoost (300 trees, depth 3, learning rate 0.05)
- Missing values were median-imputed with missing-value flags. Inputs were standardised inside each training fold (scikit-learn).
- Models were compared with 5-fold cross-validation repeated three times (15 folds) on the training set.
- A gain was "meaningful" only if ROC-AUC rose by at least 0.01 **and** a corrected repeated-CV t-test gave p < 0.05.
- Each final model was scored once on the test set, with a 1,000-sample bootstrap interval and a threshold chosen on training data (Youden's J).
- Improvement steps, added one at a time:
  1. 2009–2012 training data
  2. lifestyle variables, with age as a spline
  3. age interactions with birthplace, education and income
  4. the markers that stayed associated in the training data
  5. averaging logistic regression with XGBoost
- Also tested:
  - all 5,310 person-level NHANES 2013–2014 variables
  - 60 Optuna trials each for XGBoost and LightGBM
  - XGBoost grown from depth 2 / 300 trees to depth 10 / 5,000 trees
- Additional analyses:
  - seroprevalence with 95% Wald intervals by group
  - precision–recall curves with average precision, and calibration curves with the Brier score
  - permutation importance on the test set (20 shuffles)
  - learning curves (10–100% of each training fold)
  - Spearman correlations between the associated markers and age, foreign birth, education, income and Hispanic ethnicity

**Results: who is seropositive**
- Positive vs negative adults: mean age 55.3 vs 46.7; male 52.4% vs 46.6%; foreign-born 43.9% vs 24.6%; Mexican American or other Hispanic 32.9% vs 21.5%; less than high school 33.7% vs 20.3%; poverty–income ratio 2.2 vs 2.5; Toxocara positive 15.3% vs 5.2%; Spanish-language interview 23.9% vs 7.6%; no health insurance 27.2% vs 19.9%.
- Among U.S.-born adults, seroprevalence rose from 4.7% at 18–29 years to 24.1% at 70–80.
- Foreign-born adults were already 16.9% positive at 18–29 (30.4% at 70–80), and 21–27% whatever their time in the U.S., against 11.6% for U.S.-born adults. This points to infection acquired before migration.
- Highest: other Hispanic adults (30.4%) and adults with less than 9th-grade education (29.1%). Seroprevalence fell with income, from 18.5% below the poverty line to 10.5% at the highest income.

**Results: RQ1, baseline models**

| Model | Baseline CV / Test | + Lifestyle CV / Test | + 44 markers CV / Test |
|---|---|---|---|
| Logistic regression | 0.722 / 0.712 | 0.718 / 0.713 | 0.699 / 0.727 |
| Random forest | 0.716 / 0.711 | 0.717 / 0.697 | 0.723 / 0.711 |
| Gradient boosting | 0.711 / 0.699 | 0.704 / 0.700 | 0.705 / 0.712 |
| XGBoost | 0.715 / 0.704 | 0.713 / 0.694 | 0.708 / 0.724 |

- The CV standard deviation was 0.016–0.022.
- No model was significantly better than logistic regression, and lifestyle variables did not help.
- A single model's score varied from about 0.67 to 0.76 across the 15 folds, much more than the differences between models.

**Results: RQ2, markers**
- With all 44 markers, cross-validated ROC-AUC did not improve significantly. Logistic regression fell (0.718 to 0.699) because it over-fits many weak inputs.
- 17 of the 44 markers stayed associated after adjustment for age, sex and birthplace:
  - Spanish-language interview (OR 2.83), no health insurance (1.88), HSV-2 (1.80), not a U.S. citizen (1.61), HSV-1 (1.46), Toxocara IgG level (1.39 per SD), construction or soil work (1.30)
  - missing teeth, crowding, household size, blood lead and food insecurity (1.17–1.20 per SD); renting, cotinine and eosinophils were also among the 17
  - inversely associated: shellfish (0.79) and blood mercury (0.82 per SD)
- Most markers repeat the migration and poverty signal:
  - Spanish interview correlated with Hispanic ethnicity (ρ = 0.60) and foreign birth (0.51); non-citizenship with foreign birth (0.62)
  - missing teeth and blood lead with age (0.60 and 0.49); crowding with younger age (−0.44); food insecurity with low income (−0.47)
  - only Toxocara IgG level, HSV-2, eosinophils and construction work had |ρ| < 0.25 with all five baseline variables
- Permutation importance:
  - In XGBoost, age, Toxocara IgG level, race/ethnicity and years in the U.S. mattered most (0.010–0.051 drop in ROC-AUC); no other input exceeded 0.007.
  - Logistic regression leaned on missing-value flags. Drug use was not asked of adults aged 70+, so its missing flag acts as a stand-in for old age (21.8% positive when missing vs 13.0%).
- Average precision (prevalence 0.150): 0.333 baseline logistic regression, 0.379 with all markers, 0.368 for XGBoost, 0.375 for the improved model. All models were well calibrated; Brier score 0.116 → 0.113.

**Results: RQ3, improving the model**

| Step | CV | Test |
|---|---|---|
| 0 Baseline logistic regression | 0.723 | 0.713 |
| 1 + 2009–2012 training data | 0.727 | 0.712 |
| 2 + lifestyle, age spline | 0.726 | 0.713 |
| 3 + age interactions | 0.730 | 0.714 |
| 4 + associated markers | 0.735 | 0.733 |
| 5 + XGBoost ensemble (final) | 0.740 | 0.730 |

- Final model: CV ROC-AUC 0.740 (corrected p = 0.0002 against step 0); test ROC-AUC 0.730 (95% CI 0.688–0.774). Paired bootstrap gain over the baseline: +0.017 (95% CI 0.000–0.034), so consistent but small.
- At the threshold chosen on training data (0.136): sensitivity 70.7%, specificity 64.4% (balanced accuracy 67.6% vs 65.8% for the baseline). It found 8 more of the 150 positives (106 vs 98) at the cost of 15 more false positives. Accuracy at 0.5 was 86.1% vs 85.0% for predicting everyone negative, which shows why accuracy is not informative.
- Ceiling:
  - all 5,310 variables reached 0.721
  - Optuna-tuned XGBoost / LightGBM: CV 0.740 / 0.741, test 0.719 / 0.724
  - larger XGBoost models memorised the training data (training ROC-AUC 0.76 → 1.00) while the score on unseen adults fell (0.739 → 0.718)
  - learning curves: baseline logistic regression was nearly flat after about 1,600 adults (0.719 with 1,601, 0.724 with 3,203); models with all markers stayed below the baseline (0.700 logistic regression, 0.703 XGBoost, vs 0.724 and 0.714)
- Conclusion: the limit is the information, not the learner. NHANES asks nothing about cats, undercooked meat or soil, and IgG records infection acquired over a lifetime.

**Results: practical use**
- A ROC-AUC of 0.73 is too low to diagnose infection, but it can rank people for testing.
- At the chosen threshold, the model flagged 409 of 1,002 test adults (40.8%), which included 70.7% of all infections.
- The positive predictive value of a flag was 25.9% (1.7 times the 15.0% prevalence). A negative result left a 7.4% chance of infection (44 of 593).
- The model would need recalibration before use in other countries.

**Limitations (from the paper)**
- The data are cross-sectional and unweighted, so the models describe the NHANES sample, not the U.S. population.
- IgG lasts for life, so present-day markers describe the person now, not at infection.
- There is one held-out test set with 150 positives, so test differences below about 0.045 are noise.
- Some markers were measured only in some age groups (HSV at 18–49 years, drug use below 70 years), so their missing flags partly encode age.

# Part 4. Section-by-section instructions

**Title.** Keep the current title. Add a comment suggesting the paper's title as an option: "Machine Learning Prediction of Toxoplasma gondii Seropositivity Using NHANES Survey and Biological Marker Data".

**Background & Problem Statement.** Keep the text. Fix only the citations so they match the reference list in Part 5:
- the 31% meta-analysis is Sengupta et al. 2025
- the NHANES study of older adults (cognition) is Kruszon-Moran et al. 2023
- the NHANES and UK Biobank depression study is Gale et al. 2021

**Literature Review and Research Gap.**
- *Epidemiology:* add, from the paper, that earlier NHANES studies also linked infection with crowding and soil-related work (Jones et al. 2001), and that a case-control study of recently infected patients added raw shellfish, undercooked meat, unpasteurised goat milk and kittens (Jones et al. 2009). Keep the point that these studies describe associations, not how well individuals can be classified.
- *Health, mood and cognition:* replace "Gale et al. [4] found small decreases in cognitive processing and working memory in NHANES adults" with: "Kruszon-Moran et al. studied 2,643 NHANES adults aged 60 and over. Infection was linked to lower scores on three cognitive tests, but after adjustment the link remained for only one of them." Replace the Pearce sentence with: "By contrast, Gale et al. found no association with depression in 2,484 NHANES 2013–2014 adults and 2,271 UK Biobank participants." Change "which RQ2 tests directly" to match the new research questions.
- *Low-cost detection, ML on NHANES, questionnaire-based ML:* keep. Make sure the Ichwan et al. citation points to the Ichwan reference, not an LLM paper.
- *Research gap:* rewrite the last paragraph to match the paper. No study (a) compares standard learners for T. gondii on a fixed held-out NHANES test set, (b) screens a wide range of NHANES biological and exposure markers with adjusted associations, and (c) tests how far prediction can be pushed (more cycles, flexible age, ensembles, tuning) and explains what limits it.

**Research Questions and Objectives.** Replace the five RQs with the paper's three:
- RQ1. How well do standard machine-learning models predict seropositivity from NHANES demographic and lifestyle data?
- RQ2. Do biological and exposure markers measured in NHANES improve these models, and which markers are associated with infection?
- RQ3. How far can the models be improved with more data, better use of age, ensembling and tuning, and what limits them?

Rewrite the objectives to match:
1. a reproducible, leakage-free pipeline across 20+ NHANES files and three cycles
2. four learners on baseline and lifestyle features with repeated CV and the corrected t-test (RQ1)
3. adding 44 markers and adjusted odds ratios (RQ2)
4. the five improvement steps, tuning and model size (RQ3)
5. a single test-set evaluation per final model with bootstrap intervals, PR curves, calibration, permutation importance, learning curves and marker correlations (RQ1–RQ3)

**Proposed Methodology.** Update it to match Part 3: data and cohort (including the pooled 2009–2014 cycles for RQ3), the three feature groups, models and settings, evaluation rules, improvement steps and additional analyses. Keep my explanations of ROC-AUC, PR-AUC, the Brier score and why accuracy is not used. Replace "merges eight NHANES files" with the new scope.

**Results Summary (new section, after the Methodology).** Add a short results section, about one page, using only Part 3:
- who is seropositive (2–3 sentences)
- RQ1 with the Table III numbers as a small table
- RQ2: markers associated but not predictive, with the correlation explanation
- RQ3: the improvement-steps table, the final model and the ceiling
- practical use (flags 41%, finds 71% of infections, PPV 25.9%)

Follow the honesty rules in Part 2.

**Expected Contribution.**
- *To knowledge:* a fair comparison of four learners on a fixed test set, a screen of 44 markers with adjusted associations, and an explanation of where the prediction limit comes from (missing exposure questions, not the learner).
- *To practice:* the model can rank people for testing (flagging about 41% finds about 7 in 10 infections), it is a research tool, not a diagnostic test, and future studies should collect direct exposure questions.
- Remove the sentence about LLM text-generation errors.

**Ethical Considerations.**
- Remove the sentence that cites the LLM evaluation paper ("Evaluations of clinical AI tools warn ..."). Replace it with: "Its outputs should not drive decisions about any person without expert oversight."
- Note that all three NHANES cycles (2009–2014) are public and approved by the NCHS Research Ethics Review Board.
- Add one sentence that several markers are sensitive (HIV, drug use, citizenship, interview language) and are reported only as grouped results, with care to avoid stigma.

**Timeline.** Keep the weeks. Change "merging the 8 files" to "merging the NHANES files across three cycles". Add the marker screen to week 5 and the improvement steps to week 6.

**References.** Replace the whole list with the entries in Part 5:
- Keep only the references that are cited.
- Number them in order of first citation.
- Delete every LLM reference.

# Part 5. Reference entries to use (IEEE format; number them by first citation)

- P. P. Sengupta, S. S. Jacob, K. P. Suresh, S. Rajamani, and S. M. Maharana, "Exploring global trends in human toxoplasmosis seroprevalence by meta-analysis," Exp. Parasitol., vol. 275, Art. no. 108971, 2025, doi: 10.1016/j.exppara.2025.108971.
- J. L. Jones, D. Kruszon-Moran, H. N. Rivera, C. Price, and P. P. Wilkins, "Toxoplasma gondii seroprevalence in the United States 2009–2010 and comparison with the past two decades," Am. J. Trop. Med. Hyg., vol. 90, no. 6, pp. 1135–1139, 2014, doi: 10.4269/ajtmh.14-0013.
- J. Flegr and D. Q. Escudero, "Impaired health status and increased incidence of diseases in Toxoplasma-seropositive subjects – an explorative cross-sectional study," Parasitology, vol. 143, no. 14, pp. 1974–1989, 2016, doi: 10.1017/S0031182016001785.
- D. Kruszon-Moran, D. Brody, and B. Pearce, "Association of infection with Toxoplasma gondii and Toxocara on cognitive function among US adults aged 60 and over, NHANES 2011–2014," J. Epidemiol. Community Health, vol. 77, no. 5, pp. 315–321, 2023, doi: 10.1136/jech-2022-219772.
- S. D. Gale, L. D. Erickson, B. L. Brown, and D. W. Hedges, "Examining the relationship between Toxoplasma gondii and seropositivity and serointensity and depression in adults from the United Kingdom and the United States: A cross-sectional study," Pathogens, vol. 10, no. 9, Art. no. 1101, 2021, doi: 10.3390/pathogens10091101.
- A. Achtziger, C. Alós-Ferrer, and M. Garagnani, "An inexpensive method to measure latent toxoplasmosis and its behavioral consequences," Acta Psychol., vol. 254, Art. no. 104814, 2025, doi: 10.1016/j.actpsy.2025.104814.
- J. L. Jones, D. Kruszon-Moran, M. Wilson, G. McQuillan, T. Navin, and J. B. McAuley, "Toxoplasma gondii infection in the United States: Seroprevalence and risk factors," Am. J. Epidemiol., vol. 154, no. 4, pp. 357–365, 2001.
- J. L. Jones, V. Dargelas, J. Roberts, C. Press, J. S. Remington, and J. G. Montoya, "Risk factors for Toxoplasma gondii infection in the United States," Clin. Infect. Dis., vol. 49, no. 6, pp. 878–884, 2009.
- J. L. Jones et al., "Toxoplasma gondii infection in the United States, 2011–2014," Am. J. Trop. Med. Hyg., vol. 98, no. 2, pp. 551–557, 2018, doi: 10.4269/ajtmh.17-0677.
- J. Guo, Q. He, and Y. Li, "Machine learning-based prediction of vitamin D deficiency: NHANES 2001–2018," Front. Endocrinol., vol. 15, Art. no. 1327058, 2024, doi: 10.3389/fendo.2024.1327058.
- S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017.
- W. W. Ichwan, P. A. Tandra, N. N. Qomariyah, R. F. Meirawan, and F. Setiawan, "Machine learning-based prediction of seropositive Toxoplasma gondii from lifestyle factors," manuscript, Bina Nusantara Univ., Jakarta, Indonesia.
- B. J. Luft and J. S. Remington, "Toxoplasmic encephalitis in AIDS," Clin. Infect. Dis., vol. 15, no. 2, pp. 211–222, 1992. (Use only if you mention encephalitis in people living with HIV.)
- C. L. Johnson, S. M. Dohrmann, V. L. Burt, and L. K. Mohadjer, "National Health and Nutrition Examination Survey: Sample design, 2011–2014," Nat. Center Health Statist., Hyattsville, MD, USA, Vital Health Stat. Ser. 2, no. 162, 2014. [Online]. Available: https://stacks.cdc.gov/view/cdc/22155
- Centers for Disease Control and Prevention, National Center for Health Statistics, "National Health and Nutrition Examination Survey data, 2009–2014," Hyattsville, MD, USA. [Online]. Available: https://wwwn.cdc.gov/nchs/nhanes/
- Centers for Disease Control and Prevention, National Center for Health Statistics, "National Health and Nutrition Examination Survey 2013–2014 data documentation: Toxoplasma gondii antibodies (SSTOXO_H)." [Online]. Available: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SSTOXO_H.htm (accessed Oct. 19, 2026).
- Centers for Disease Control and Prevention, National Center for Health Statistics, "National Health and Nutrition Examination Survey 2013–2014 data documentation: Demographic variables and sample weights (DEMO_H)." [Online]. Available: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm (accessed Oct. 19, 2026).
- K. Kroenke, R. L. Spitzer, and J. B. W. Williams, "The PHQ-9: Validity of a brief depression severity measure," J. Gen. Intern. Med., vol. 16, no. 9, pp. 606–613, 2001, doi: 10.1046/j.1525-1497.2001.016009606.x.
- Centers for Disease Control and Prevention, National Center for Health Statistics, "National Health and Nutrition Examination Survey 2013–2014 data documentation: Toxocara antibodies (SSTOCA_H)." [Online]. Available: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SSTOCA_H.htm (accessed Oct. 19, 2026).
- S. R. Healy, E. R. Morgan, J. M. Prada, and M. Betson, "Brain food: rethinking food-borne toxocariasis," Parasitology, vol. 149, no. 1, pp. 1–9, 2022, doi: 10.1017/S0031182021001591.
- E. W. Liu, E. S. Elder, H. N. Rivera, D. Kruszon-Moran, S. Handali, and J. L. Jones, "Concurrent seroprevalence of antibodies to Toxoplasma gondii and Toxocara species in the United States, 2011–2014," Clin. Infect. Dis., 2018, doi: 10.1093/cid/ciy729.
- W. McKinney, "Data structures for statistical computing in Python," in Proc. 9th Python Sci. Conf., 2010, pp. 56–61, doi: 10.25080/Majora-92bf1922-00a.
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.
- T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.
- G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 3146–3154.
- T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019, pp. 2623–2631.
- L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.
- J. H. Friedman, "Greedy function approximation: A gradient boosting machine," Ann. Statist., vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.
- T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.
- C. Nadeau and Y. Bengio, "Inference for the generalization error," Mach. Learn., vol. 52, pp. 239–281, 2003, doi: 10.1023/A:1024068626366.
- N. F. A. Polanunu, S. Wahyuni, and F. Hamid, "Seroprevalence and associated risk factors of Toxoplasma gondii infection among pregnant mother in Makassar, Indonesia," PLoS ONE, vol. 16, no. 6, Art. no. e0245572, 2021, doi: 10.1371/journal.pone.0245572.
- Centers for Disease Control and Prevention, National Center for Health Statistics, "Ethics Review Board approval." [Online]. Available: https://www.cdc.gov/nchs/nhanes/about/erb.html (accessed Oct. 19, 2026).

Do not use these old references any more: Montoya & Liesenfeld 2004, Gale et al. 2015, Pearce et al. 2012, Jones et al. 2007, Krueger et al. 2014, Farmer et al. 2017, and all LLM papers (Singhal, Sandmann, Wu, Chimirri, Wang, Hendrix, Asgari, Lobentanzer).

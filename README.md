# SECOM semiconductor pass/fail prediction - reproducible course project

## Headline decision
**Would a process/quality engineer send a production entity for additional inspection?** This is an advisory flag, NOT automatic scrapping or customer release. `Fail` = 1; `Pass` = 0.

## Reproduce exactly
From the repository folder, with Python 3.10+:

```bash
python -m pip install -r requirements.txt
python run_project.py
```

The script uses **seed 42, stratified 70/30 split, 471 test rows**, and the frozen checkpoint model definitions. It downloads the raw data from UCI if not present in `data/`. If the network download fails, manually place `secom.data` and `secom_labels.data` in `data/`, then rerun the exact command. **This download cannot be verified in the present offline environment.**

To run the independent code/math check without downloading data:

```bash
python run_project.py --smoke-test
```

To examine transparent 3-fold hyperparameter search using **training rows only**:

```bash
python run_project.py --tune
```

This creates `outputs/training_only_tuning.csv`; it does **not** retroactively change the frozen headline or pretend tuned results were evaluated on held-out test data. Updating the final headline with newly tuned models requires a new evaluation and matching report revision.

## Inputs and outputs
- Input: UCI SECOM, `secom.data` (1567 x 590) and `secom_labels.data` (-1 = pass, +1 = fail, timestamps).
- Printed headline: `outputs/model_metrics.csv` (accuracy, fail recall with 95% Wilson CI, false positives/negatives), `outputs/slice_metrics.csv`, `outputs/illustrative_costs.csv`.
- Figures: `figures/failure_recall.png`, `figures/slice_recall.png`, `figures/train_PCA.png`.
- Config: `outputs/reproduction.json` holds split parameters and PCA explained variance.

## Why we chose these settings
The 70/30 stratified split and seed 42 were committed in the project gate. **Failure recall** is the primary score since the company loses money when faulty production goes undetected. Accuracy can be misleading with 93% Pass labels. The predeclared baseline always predicts Pass. Logistic Regression (linear and relatively interpretable) and a Decision Tree (nonlinear rules, reasonably interpretable) were the checkpoint's two models. Both use median imputation based on TRAINING rows only; logistic regression additionally standardizes training features. Class weights address class imbalance.

**Important distinction:** The checkpoint's hyperparameters (`C=0.1` for logistic regression; `max_depth=5`, `min_samples_leaf=10` for the tree) were fixed before this report, but **were not selected by a documented train-only tuning search**. The optional `--tune` command now performs such a search for audit transparency; the PDF cannot claim those scores without actually rerunning it.

## Unsupservised analysis and worst slice
- A **label-free, training-feature-derived** quality grouping uses the median number of missing sensors in the training samples (30). It yields `<=30` vs `>30` missing readings. Outcomes are used only to evaluate group-specific recall after the split is set. This is exploratory segmentation, *not* evidence of causal manufacturing mechanisms.
- `figures/train_PCA.png` explores 2D PCA on standardized training features alone. Labels are applied only for plot colors AFTER fitting PCA. PCA does not trigger automatic decisions and does not identify physical failure causes.
- In the earlier checkpoint output (not freshly rerun in this environment), the worst observed slice was **logistic regression; >30 missing**: **n=220, 13 actual failures, 12 missed**, fail recall 1/13 = 7.7%. Wilson 95% CI ~1.4-33.3%. This is a small and imprecise subgroup.

## Data and business limitations
- UCI explicitly calls each row a *production entity*, not guaranteed to be a single finished chip.
- Labels represent in-house pass/fail testing, not customer failures; this is a hypothetical early-warning action.
- SECOM data are historic and anonymized. We do not know sensor identities, production timing at decision, site drift, true cost per inspection, or consequences in dollars.
- A random stratified test can overstate future performance relative to chronological/out-of-factory validation.
- The report's economic ratios are illustrative break-even thresholds, not real observed company accounting figures.

## Submission
Submit `SECOM_Final_Project_Report.pdf` plus the repository ZIP if a public repository cannot be supplied. PDF first page identifies its accompanying ZIP. Also provide references and the AI-use appendix in the report. The PDF currently reflects the earlier checkpoint's verified outputs; rerun with the official data before claiming final reproduction. Do not invent CV tuning results.

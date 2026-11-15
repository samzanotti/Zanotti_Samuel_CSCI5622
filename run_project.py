"""SECOM final project | run with: python run_project.py

0 = pass, 1 = fail. A prediction of fail means: flag for inspection.
The two classification models are tuned using training data ONLY.
"""
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------- settings (same split and metric as the proposal) ----------
SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 3
MISSED_FAILURE_COST = 10   # example only: 10 times an unnecessary inspection
EXTRA_INSPECTION_COST = 1  # example cost units, NOT company dollar amounts
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'
DATA_URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'
# ---------------------------------------------------------------------


def load_secom():
    """Read SECOM, downloading the two UCI files if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    names = ['secom.data', 'secom_labels.data']

    if not all((DATA_DIR / name).exists() for name in names):
        print('Downloading the UCI SECOM dataset ...')
        try:
            with urlopen(DATA_URL, timeout=60) as response:
                zipped_data = response.read()
            with zipfile.ZipFile(BytesIO(zipped_data)) as archive:
                for name in names:
                    match = next((item for item in archive.namelist()
                                  if Path(item).name == name), None)
                    if match is None:
                        raise FileNotFoundError(f'{name} not found in ZIP')
                    (DATA_DIR / name).write_bytes(archive.read(match))
        except Exception as error:
            raise RuntimeError(
                'Download failed. Get secom.data and secom_labels.data from '
                'https://archive.ics.uci.edu/dataset/179/secom '
                'and put them in the data folder next to this script.'
            ) from error

    X = pd.read_csv(DATA_DIR / 'secom.data', sep=r'\s+', header=None)
    labels = pd.read_csv(DATA_DIR / 'secom_labels.data', sep=r'\s+', header=None)
    if X.shape != (1567, 590) or len(labels) != len(X) or labels.shape[1] != 2:
        raise ValueError('Unexpected SECOM data size or label format; check the files.')

    # UCI uses -1 for passing and +1 for failing.
    if set(labels.iloc[:, 0].unique()) != {-1, 1}:
        raise ValueError('Expected SECOM pass/fail labels of -1 and +1.')
    y = (labels.iloc[:, 0] == 1).astype(int)

    # Date and time are used ONLY for one failure-analysis slice, not as inputs.
    timestamps = pd.to_datetime(
        labels.iloc[:, 1], format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    if timestamps.isna().any():
        raise ValueError('Some SECOM timestamps could not be read.')

    X.columns = [f'sensor_{i + 1}' for i in range(X.shape[1])]
    print(f'Data: {len(X)} samples, {X.shape[1]} sensor columns, '
          f'{(y == 0).sum()} pass, {(y == 1).sum()} fail')
    return X, y, timestamps


def recall_interval(found, total):
    """95% Wilson interval for failure recall (useful for small slices)."""
    if total == 0:
        return np.nan, np.nan
    z = 1.96
    p = found / total
    d = 1 + z**2 / total
    middle = (p + z**2 / (2 * total)) / d
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / d
    return round(100 * (middle - margin), 1), round(100 * (middle + margin), 1)


# 1. load the data and make the same 70/30 split used in the proposal
X, y, timestamps = load_secom()
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
train_rows, test_rows = train_test_split(
    np.arange(len(X)), test_size=TEST_SIZE, random_state=SEED, stratify=y
)
X_train, X_test = X.iloc[train_rows], X.iloc[test_rows]
y_train, y_test = y.iloc[train_rows], y.iloc[test_rows]
print(f'Training: {len(X_train)}; testing: {len(X_test)}; '
      f'test failures: {int(y_test.sum())}')

# 2. baseline: predict pass for every test sample
predictions = {'Always Pass (baseline)': np.zeros(len(y_test), dtype=int)}

# 3. train and tune both models using 3-fold cross-validation on TRAINING data
# Imputer and scaler are inside the pipeline to avoid data leakage.
models = {
    'Logistic Regression': (
        make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(class_weight='balanced', solver='liblinear',
                               max_iter=2000, random_state=SEED)
        ),
        {'logisticregression__C': [0.01, 0.1, 1.0]}
    ),
    'Decision Tree': (
        make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            DecisionTreeClassifier(class_weight='balanced', random_state=SEED)
        ),
        {'decisiontreeclassifier__max_depth': [3, 5, 8],
         'decisiontreeclassifier__min_samples_leaf': [5, 10]}
    )
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
tuning_rows = []

for name, (model, choices) in models.items():
    search = GridSearchCV(model, choices, cv=cv, scoring='recall', n_jobs=1)
    search.fit(X_train, y_train)
    predictions[name] = search.predict(X_test)
    settings = ', '.join(f'{key.split("__")[-1]}={value}'
                         for key, value in search.best_params_.items())
    tuning_rows.append({
        'Model': name,
        'best_settings': settings,
        'training_CV_failure_recall_pct': round(100 * search.best_score_, 1)
    })
    print(f'{name}: {settings}; training CV failure recall = '
          f'{100 * search.best_score_:.1f}%')

pd.DataFrame(tuning_rows).to_csv(RESULTS_DIR / 'tuning.csv', index=False)

# 4. compare the baseline and both models on the SAME held-out test set
score_rows = []
for name, predicted in predictions.items():
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    score_rows.append({
        'Model': name, 'test_samples': len(y_test),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'accuracy_pct': round(100 * accuracy_score(y_test, predicted), 1),
        'failure_recall_pct': round(100 * recall_score(y_test, predicted,
                                                      zero_division=0), 1),
        'failure_precision_pct': round(100 * precision_score(y_test, predicted,
                                                            zero_division=0), 1)
    })
scores = pd.DataFrame(score_rows)
scores.to_csv(RESULTS_DIR / 'model_results.csv', index=False)
print('\nTEST RESULTS\n', scores.to_string(index=False))

# the model comparison plot is created with the other report figures in section 8.

# 5. PCA: look for patterns WITHOUT using labels to find the components
# Colors are added after PCA is fitted, just to see where failures appear.
X_scaled = make_pipeline(
    SimpleImputer(strategy='median', keep_empty_features=True),
    StandardScaler()
).fit_transform(X_train)
pca = PCA(n_components=2, random_state=SEED)
components = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 4))
for value, label in [(0, 'Pass'), (1, 'Fail')]:
    mask = y_train.to_numpy() == value
    plt.scatter(components[mask, 0], components[mask, 1],
                s=13, alpha=0.55, label=label)
plt.xlabel(f'PC1: {100 * pca.explained_variance_ratio_[0]:.1f}% of variance')
plt.ylabel(f'PC2: {100 * pca.explained_variance_ratio_[1]:.1f}% of variance')
plt.title('PCA of training sensor measurements')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pca.png', dpi=160)
plt.close()

# 6. failure analysis with TWO variables: missing readings and production time
# Both cutoffs are estimated using TRAINING data only.
missing_cutoff = X_train.isna().sum(axis=1).median()
time_cutoff = timestamps.iloc[train_rows].median()

missing_groups = np.where(X_test.isna().sum(axis=1) > missing_cutoff,
                          'More missing', 'Fewer missing')
time_groups = np.where(timestamps.iloc[test_rows] > time_cutoff,
                       'Later', 'Earlier')
slice_variables = {
    'Missing sensor readings': missing_groups,
    'Production timestamp': time_groups
}

slice_rows = []
for variable, groups in slice_variables.items():
    for name in ['Logistic Regression', 'Decision Tree']:
        predicted = predictions[name]
        actual = y_test.to_numpy()
        for group in np.unique(groups):
            mask = groups == group
            n = int(mask.sum())
            failures = int((actual[mask] == 1).sum())
            found = int(((actual[mask] == 1) & (predicted[mask] == 1)).sum())
            missed = failures - found
            lower, upper = recall_interval(found, failures)
            slice_rows.append({
                'Model': name, 'variable': variable, 'slice': group,
                'sample_size': n, 'actual_failures': failures,
                'detected_failures': found, 'missed_failures': missed,
                'failure_recall_pct': round(100 * found / failures, 1)
                                      if failures else np.nan,
                'recall_95pct_low': lower, 'recall_95pct_high': upper,
                'exploratory': 'Yes' if variable == 'Production timestamp' else 'No'
            })

slices = pd.DataFrame(slice_rows)
# The worst result is discovered by looking at test data, so flag it exploratory.
valid = slices['failure_recall_pct'].notna()
worst_index = slices.loc[valid, 'failure_recall_pct'].idxmin()
slices['worst_observed'] = 'No'
slices.loc[worst_index, 'worst_observed'] = 'Yes (exploratory)'
slices.loc[worst_index, 'exploratory'] = 'Yes'
slices.to_csv(RESULTS_DIR / 'slices.csv', index=False)
print('\nSLICE RESULTS\n', slices.to_string(index=False))
print(f'\nMissing-data split: >{missing_cutoff:g} vs <={missing_cutoff:g} readings')
print(f'Time split: after vs on/before {time_cutoff}')

# 7. business impact: compare errors using an ILLUSTRATIVE cost assumption
# The dataset does NOT contain actual inspection or defect cost figures.
costs = scores[['Model', 'FP', 'FN', 'TP']].copy()
costs['illustrative_error_cost'] = (costs['FP'] * EXTRA_INSPECTION_COST +
                                    costs['FN'] * MISSED_FAILURE_COST)
costs['break_even_FN_to_FP_ratio_vs_baseline'] = costs.apply(
    lambda row: round(row['FP'] / row['TP'], 2) if row['TP'] else np.nan, axis=1
)
costs.to_csv(RESULTS_DIR / 'cost_comparison.csv', index=False)
preferred = costs.loc[costs['illustrative_error_cost'].idxmin(), 'Model']

summary = (
    'Decision: flag manufacturing samples for extra inspection, not automatic scrap.\n'
    f'Locked split: 70/30 stratified, seed {SEED}; primary metric: failure recall.\n'
    'Locked baseline: always predict Pass.\n'
    'Both models tuned with TRAINING-ONLY 3-fold CV on failure recall.\n'
    f'PCA: training-only two-component view, explaining '
    f'{100 * pca.explained_variance_ratio_.sum():.1f}% of total sensor variance.\n'
    f'Slice variables: missing readings (training median {missing_cutoff:g}) '
    f'and timestamp (training median {time_cutoff}).\n'
    f'Worst observed slice (exploratory): {slices.loc[worst_index, "Model"]}, '
    f'{slices.loc[worst_index, "variable"]}, {slices.loc[worst_index, "slice"]}; '
    f'n={slices.loc[worst_index, "sample_size"]}, '
    f'failures={slices.loc[worst_index, "actual_failures"]}.\n'
    f'Example cost assumption: one missed failure = {MISSED_FAILURE_COST} '
    f'units; one unnecessary inspection = {EXTRA_INSPECTION_COST} unit.\n'
    f'Under that ASSUMPTION, lowest test-set error cost: {preferred}.\n'
    'Recommendation: at most pilot as an inspection aid; do not auto-reject chips.\n'
    'Reversal: change the recommendation if actual cost ratios, independent '
    'production validation, or additional false-positive burden disagree.\n'
    'These are production entities, not verified individual finished chips.\n'
)
(RESULTS_DIR / 'summary.txt').write_text(summary, encoding='utf-8')
print('\nBUSINESS COST COMPARISON (illustrative)\n', costs.to_string(index=False))
print('\n' + summary)
# 8. create every figure used in the final report
# All numbers below come from the dataset or the test results above.
# The colors and labels match the graphs in the Word/PDF report.
TEAL = '#21858d'
ORANGE = '#BE713B'
GREY = '#5B697A'
DARK = '#17243A'
BASELINE_COLOR = '#667789'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False
})

# figure 1: number of passing and failing production entities
counts = [(y == 0).sum(), (y == 1).sum()]
fig, ax = plt.subplots(figsize=(6.8, 1.65))
ax.barh([1, 0], counts, color=[TEAL, ORANGE], height=0.53)
for row, count in zip([1, 0], counts):
    ax.text(count + 22, row, f'{count:,} ({count / len(y):.1%})',
            va='center', color=GREY)
ax.set_yticks([1, 0], ['Pass', 'Fail'])
ax.set_xlim(0, max(counts) * 1.19)
ax.set_xlabel('Number of SECOM production entities')
ax.tick_params(axis='y', length=0)
fig.tight_layout(pad=0.65)
fig.savefig(FIGURES_DIR / 'class_balance.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# figure 2: compare failure recall for the baseline and two tuned models
fig, ax = plt.subplots(figsize=(6.8, 2.02))
recalls = scores['failure_recall_pct'].tolist()
names = ['Always Pass', 'Logistic Regression', 'Decision Tree']
ax.bar(range(3), recalls, color=[BASELINE_COLOR, TEAL, ORANGE], width=0.58)
ax.set_ylim(0, max(88, max(recalls) + 12))
ax.set_ylabel('Failure recall (%)')
ax.set_xticks(range(3), names)
for i, recall in enumerate(recalls):
    ax.text(i, recall + 2.3, f'{recall:.1f}%', ha='center',
            fontweight='bold', color=DARK)
ax.text(0.015, 0.95,
        f'Same {len(y_test)} test entities; {int(y_test.sum())} actual failures',
        transform=ax.transAxes, va='top', color=GREY, fontsize=8)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES_DIR / 'final_recall.png', dpi=170, bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'model_recall.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# figure 3: failure recall by the two variables, with group sample sizes
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.13), sharey=True)
for ax, variable, groups in [
    (axes[0], 'Missing sensor readings', ['Fewer missing', 'More missing']),
    (axes[1], 'Production timestamp', ['Earlier', 'Later'])
]:
    x = np.arange(2)
    for model_name, position, color in [
        ('Logistic Regression', -0.175, TEAL),
        ('Decision Tree', 0.175, ORANGE)
    ]:
        values = [float(slices.loc[
            (slices['Model'] == model_name) &
            (slices['variable'] == variable) &
            (slices['slice'] == group), 'failure_recall_pct'
        ].iloc[0]) for group in groups]
        ax.bar(x + position, values, width=0.35,
               color=color, label=model_name)
    labels = []
    for group in groups:
        part = slices.loc[(slices['Model'] == 'Logistic Regression') &
                          (slices['variable'] == variable) &
                          (slices['slice'] == group)].iloc[0]
        labels.append(f'{group}\n(n={int(part["sample_size"])}; '
                      f'{int(part["actual_failures"])} fail)')
    ax.set_xticks(x, labels)
    ax.set_title(variable, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.13)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Failure recall (%)')
fig.legend(axes[0].get_legend_handles_labels()[0],
           axes[0].get_legend_handles_labels()[1],
           loc='lower center', bbox_to_anchor=(0.5, -0.09),
           ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.9)
fig.savefig(FIGURES_DIR / 'slices_new.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# figure 4: hypothetical cost comparison across different cost ratios
# Let one extra inspection cost 1 unit; a missed failure costs ratio units.
ratios = np.linspace(0, 40, 401)
assumed_ratio = MISSED_FAILURE_COST / EXTRA_INSPECTION_COST
fig, ax = plt.subplots(figsize=(6.8, 2.02))
for model_name, color in [
    ('Always Pass (baseline)', BASELINE_COLOR),
    ('Logistic Regression', TEAL),
    ('Decision Tree', ORANGE)
]:
    row = scores.loc[scores['Model'] == model_name].iloc[0]
    values = int(row['FP']) + int(row['FN']) * ratios
    label = 'Always Pass' if model_name.startswith('Always') else model_name
    ax.plot(ratios, values, label=label, color=color, linewidth=2)
    ax.scatter([assumed_ratio],
               [int(row['FP']) + int(row['FN']) * assumed_ratio],
               color=color, s=24, zorder=4)
ax.axvline(assumed_ratio, color=GREY, linestyle='--', linewidth=1)
ax.text(assumed_ratio + 0.6, 60,
        f'Assumed ratio = {assumed_ratio:g}:1', color=GREY, fontsize=7)
ax.set_ylabel('Illustrative total error cost\n(relative units)')
ax.set_xlabel('Cost of missed failure / cost of unnecessary inspection')
ax.set_xlim(0, 40)
ax.set_ylim(0, 1300)
ax.legend(ncol=3, fontsize=7, frameon=False, loc='upper left')
ax.grid(alpha=0.12)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES_DIR / 'cost_new.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# extra: a graphic version of the report's cost-based confusion-matrix TABLE
# This table explains the four possible decisions; it is not model output.
fig, ax = plt.subplots(figsize=(10.5, 2.4))
ax.axis('off')
cell_text = [
    ['Pass', 'False positive: unnecessary inspection,\nlabor, tool capacity, and delay',
     'True negative: correct pass,\nno extra model-triggered inspection'],
    ['Fail', 'True positive: failure correctly\nflagged for review',
     'False negative: failed entity receives\nno model flag']
]
headings = ['Actual state', 'Predicted Fail: inspect',
            'Predicted Pass: no model flag']
table = ax.table(cellText=cell_text, colLabels=headings,
                 colWidths=[0.14, 0.43, 0.43],
                 cellLoc='left', colLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.0)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#C7CED3')
    if row == 0:
        cell.set_facecolor('#EAF1F2')
        cell.set_text_props(weight='bold')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'decision_cost_matrix.png',
            dpi=170, bbox_inches='tight')
plt.close(fig)

print('Saved results/*.csv and results/summary.txt')
print('Saved report figures: class_balance.png, final_recall.png, '
      'slices_new.png, cost_new.png, decision_cost_matrix.png')
print('Saved additional figures: model_recall.png and pca.png')


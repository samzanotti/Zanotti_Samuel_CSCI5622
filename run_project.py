"""SECOM final-project reproduction. Run: python run_project.py

The printed headline numbers use the FIXED checkpoint hyperparameters, not
post-hoc values chosen to maximize the held-out test score.

Optional: python run_project.py --tune
Performs training-only 3-fold CV searches and SAVES THEIR RESULTS SEPARATELY.
Do not paste tuned-model figures into the PDF without regenerating the report.
"""
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
import argparse
import json
import zipfile
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'
SEED = 42
TEST_SIZE = 0.30


def load_data():
    DATA.mkdir(exist_ok=True)
    needed = ('secom.data', 'secom_labels.data')
    if not all((DATA / file).exists() for file in needed):
        print('Downloading SECOM from UCI ...')
        try:
            with urlopen(URL, timeout=60) as response:
                raw = response.read()
            with zipfile.ZipFile(BytesIO(raw)) as z:
                for name in needed:
                    entry = next((s for s in z.namelist() if Path(s).name == name), None)
                    if entry is None:
                        raise FileNotFoundError(f'{name} missing from UCI archive')
                    (DATA / name).write_bytes(z.read(entry))
        except Exception as err:
            raise RuntimeError('Download failed. Manually place secom.data and '
                               'secom_labels.data in the data/ folder. '
                               'See https://archive.ics.uci.edu/dataset/179/secom') from err
    X = pd.read_csv(DATA / 'secom.data', sep=r'\s+', header=None)
    label_file = pd.read_csv(DATA / 'secom_labels.data', sep=r'\s+', header=None)
    y = (label_file.iloc[:, 0].astype(int) == 1).astype(int)
    assert X.shape == (1567, 590), f'Unexpected X shape: {X.shape}'
    assert y.value_counts().to_dict() == {0: 1463, 1: 104}
    X.columns = [f'sensor_{j+1}' for j in range(X.shape[1])]
    return X, y


def model_definitions():
    # Same fixed settings as simplified checkpoint (not claimed to be tuned).
    return {
        'Logistic Regression': make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(class_weight='balanced', C=0.1,
                               solver='liblinear', max_iter=3000)),
        'Decision Tree': make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            DecisionTreeClassifier(class_weight='balanced', max_depth=5,
                                   min_samples_leaf=10, random_state=SEED)),
    }


def wilson_interval(successes, total):
    """Approximate 95% Wilson interval for a binomial fraction."""
    if total == 0:
        return (float('nan'), float('nan'))
    z = 1.96
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    margin = z * sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return (max(0.0, (center-margin)*100), min(100.0, (center+margin)*100))


def score_predictions(actual, pred, name):
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    low, high = wilson_interval(int(tp), int(tp+fn))
    return dict(Model=name, N=len(actual), TN=int(tn), FP=int(fp), FN=int(fn),
                TP=int(tp), accuracy_pct=round(100*accuracy_score(actual, pred), 2),
                fail_recall_pct=round(100*recall_score(actual, pred, zero_division=0), 2),
                fail_precision_pct=round(100*precision_score(actual, pred, zero_division=0), 2),
                recall_CI_low=round(low, 1), recall_CI_high=round(high, 1))


def make_slices(X_train, X_test, y_test, pred_by_model):
    # Training feature-missingness median; no test labels used to make the groups.
    cutoff = float(X_train.isna().sum(axis=1).median())
    groups = np.where(X_test.isna().sum(axis=1).to_numpy() > cutoff,
                      'Over 30 missing' if cutoff == 30 else 'More missing',
                      '30 or fewer missing' if cutoff == 30 else 'Fewer missing')
    yy = y_test.to_numpy()
    table = []
    for name, pred in pred_by_model.items():
        for group in np.unique(groups):
            mask = groups == group
            failures = int((yy[mask] == 1).sum())
            missed = int(((yy[mask] == 1) & (pred[mask] == 0)).sum())
            found = failures - missed
            low, high = wilson_interval(found, failures)
            table.append(dict(Model=name, Slice=group, n=int(mask.sum()),
                              actual_failures=failures, missed_failures=missed,
                              found_failures=found,
                              failure_recall_pct=round(100*found/failures, 1) if failures else None,
                              recall_CI_low=round(low, 1) if failures else None,
                              recall_CI_high=round(high, 1) if failures else None))
    return cutoff, pd.DataFrame(table)


def make_figures(metrics, slices, X_train, y_train):
    plt.figure(figsize=(7, 3.3))
    plt.bar(metrics['Model'], metrics['fail_recall_pct'])
    plt.ylabel('Recall on actual failures (%)')
    plt.ylim(0, 55)
    plt.xticks(rotation=11, ha='right')
    plt.tight_layout()
    plt.savefig(FIG / 'failure_recall.png', dpi=170)
    plt.close()

    plt.figure(figsize=(7, 3.3))
    for name in slices.Model.unique():
        s = slices[slices.Model == name]
        plt.plot(s['Slice'], s['failure_recall_pct'], marker='o', label=name)
    plt.ylabel('Failure recall in slice (%)')
    plt.ylim(0, 70)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / 'slice_recall.png', dpi=170)
    plt.close()

    # Exploratory, UNSUPERVISED PCA on training features only. The labels are
    # applied solely as plot colors after PCA; not to fit PCA or select PCs.
    prep = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler())
    scaled = prep.fit_transform(X_train)
    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(scaled)
    plt.figure(figsize=(7, 3.5))
    for val, label in [(0, 'Pass'), (1, 'Fail')]:
        mask = y_train.to_numpy() == val
        plt.scatter(pcs[mask, 0], pcs[mask, 1], alpha=.6, s=13, label=f'{label}: {mask.sum()}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} explained variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} explained variance)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(FIG / 'train_PCA.png', dpi=170)
    plt.close()
    return pca.explained_variance_ratio_.tolist()


def compare_costs(metrics):
    # Costs are in units of one unnecessary inspection, not actual dollars.
    rows = []
    for cost_ratio in (1, 3, 5, 7, 10, 20):
        row = {'FN_to_FP_cost_ratio': cost_ratio}
        for m in metrics.to_dict('records'):
            row[m['Model']] = m['FN']*cost_ratio + m['FP']
        rows.append(row)
    return pd.DataFrame(rows)


def tune_on_training_only(X_train, y_train):
    """Optional TRAIN-ONLY CV; does not alter frozen checkpoint headline."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    searches = {
        'Logistic Regression': (model_definitions()['Logistic Regression'],
                                {'logisticregression__C': [0.01, 0.1, 1.0]}),
        'Decision Tree': (model_definitions()['Decision Tree'],
                          {'decisiontreeclassifier__max_depth': [3, 5, 8],
                           'decisiontreeclassifier__min_samples_leaf': [5, 10]}),
    }
    rows = []
    for name, (pipeline, params) in searches.items():
        grid = GridSearchCV(pipeline, params, scoring='recall', cv=cv, n_jobs=1,
                            refit=True, return_train_score=False)
        grid.fit(X_train, y_train)
        rows.append({'Model': name, 'best_CV_failure_recall': grid.best_score_,
                     'best_training_only_params': json.dumps(grid.best_params_)})
    return pd.DataFrame(rows)


def smoke_test():
    actual = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    r = score_predictions(actual, pred, 'test')
    assert (r['TN'], r['FP'], r['FN'], r['TP']) == (1, 1, 1, 1)
    assert r['fail_recall_pct'] == 50.0
    low, high = wilson_interval(1, 2)
    assert 0 <= low <= 50 <= high <= 100
    print('SMOKE TEST PASS: confusion labels, recall and CI computation')


def main(tune):
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    predictions = {'Always Pass (baseline)': np.zeros(len(y_test), dtype=int)}
    for name, model in model_definitions().items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
    metrics = pd.DataFrame([score_predictions(y_test, pred, name)
                            for name, pred in predictions.items()])
    cutoff, slices = make_slices(X_train, X_test, y_test,
                                {k:v for k,v in predictions.items() if 'baseline' not in k})
    costs = compare_costs(metrics)
    variance = make_figures(metrics, slices, X_train, y_train)
    metrics.to_csv(OUT / 'model_metrics.csv', index=False)
    slices.to_csv(OUT / 'slice_metrics.csv', index=False)
    costs.to_csv(OUT / 'illustrative_costs.csv', index=False)
    (OUT / 'reproduction.json').write_text(json.dumps(dict(
        seed=SEED, test_size=TEST_SIZE, train_n=len(y_train), test_n=len(y_test),
        test_failures=int(y_test.sum()), missingness_cutoff=cutoff,
        PCA_variance_explained_train=variance), indent=2))
    print('\nHELD-OUT RESULTS (LOCKED CHECKPOINT MODELS)\n', metrics.to_string(index=False))
    print('\nSLICES\n', slices.to_string(index=False))
    print('\nSaved outputs/*.csv and figures/*.png')
    if tune:
        print('\nRunning 3-fold CV on TRAINING ONLY; results saved separately...')
        tuned = tune_on_training_only(X_train, y_train)
        tuned.to_csv(OUT / 'training_only_tuning.csv', index=False)
        print(tuned.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true',
                        help='run training-only CV for each model, separate from locked headline')
    parser.add_argument('--smoke-test', action='store_true',
                        help='check math without downloading data')
    args = parser.parse_args()
    if args.smoke_test: smoke_test()
    else: main(args.tune)

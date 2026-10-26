"""
Shared pieces for the Makassar experiments, so every script uses exactly the same setup:
data loading, feature sets, preprocessing, cross-validation folds, the corrected t-test,
and a small PyTorch MLP that can learn from soft labels (needed for self-distillation).

Run the experiment scripts from inside makassar_project/.
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

torch.set_num_threads(1)     # tiny data: one thread is faster and fully repeatable

DATA_CSV = os.path.join('data', 'processed', 'makassar_clean.csv')
OUT_DIR = 'outputs'

# Background: who the woman is (no exposure information)
BACKGROUND = ['age', 'trimester', 'gravidity', 'parity', 'education', 'abortion_history', 'employed']
# Exposures: the direct transmission questions (yes/no) + water sources
EXPOSURE_BINARY = ['raises_cat', 'cat_contact', 'cleans_litter', 'soil_contact', 'eats_chicken_satay',
                   'eats_raw_vegetables', 'drinks_unboiled_water', 'drinks_gallon_water']
CATEGORICAL = ['washing_water', 'cooking_water']

FEATURE_SETS = {
    'A_background': BACKGROUND,
    'B_exposures': EXPOSURE_BINARY + CATEGORICAL,
    'C_all': BACKGROUND + EXPOSURE_BINARY + CATEGORICAL,
}


def load_data():
    """Clean data without the 2 borderline IgG results (the original paper excluded them)."""
    df = pd.read_csv(DATA_CSV)
    return df[df['igg_borderline'] == 0].reset_index(drop=True)


def make_preprocessor(features):
    """Scale number columns, one-hot encode the water-source columns. Fitted inside each training fold."""
    cat = [c for c in CATEGORICAL if c in features]
    num = [c for c in features if c not in cat]
    parts = [('num', StandardScaler(), num)]
    if cat:
        parts.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat))
    return ColumnTransformer(parts)


def make_cv_splits(X, y, n_repeats=20, seed=42):
    """Stratified 5-fold CV repeated n_repeats times. Made once, so every model gets the same folds."""
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(X, y))


def corrected_ttest(base_scores, other_scores, test_train_ratio=0.25):
    """Nadeau-Bengio corrected resampled t-test on per-fold scores. Returns (mean difference, p-value)."""
    diff = np.asarray(other_scores) - np.asarray(base_scores)
    k = len(diff)
    t = diff.mean() / np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    return diff.mean(), 2 * stats.t.sf(abs(t), df=k - 1)


class TorchMLP(ClassifierMixin, BaseEstimator):     # this order: sklearn reads it as a classifier
    """Small neural network (one hidden layer) trained with binary cross-entropy.

    y may be hard labels (0/1) or soft labels (any value between 0 and 1), so the same model
    can be a normal baseline or a self-distillation student.
    """

    def __init__(self, hidden=16, weight_decay=1e-3, lr=0.01, epochs=300, random_state=0):
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        target = torch.as_tensor(np.asarray(y, dtype=np.float32))
        torch.manual_seed(self.random_state)
        self.net_ = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):          # full-batch training: the data set is tiny
            optimiser.zero_grad()
            loss = loss_fn(self.net_(X).squeeze(1), target)
            loss.backward()
            optimiser.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.as_tensor(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SoftLabelLogReg(ClassifierMixin, BaseEstimator):
    """Logistic regression that also accepts soft labels.

    Duplicate-row trick: every person appears twice, once labelled 1 with weight y and once
    labelled 0 with weight 1 - y. The weighted loss equals cross-entropy with target y, and the
    weights per person still add up to 1, so C means the same as in normal logistic regression.
    With hard labels (0/1) this is exactly normal logistic regression.
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X2 = np.vstack([X, X])
        y2 = np.r_[np.ones(len(y)), np.zeros(len(y))]
        w2 = np.r_[y, 1 - y]
        self.model_ = LogisticRegression(C=self.C, max_iter=5000).fit(X2, y2, sample_weight=w2)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibration_slope(y, p):
    """Slope of a logistic fit of y on logit(p): 1 = ideal, below 1 = predictions too extreme."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(penalty=None, max_iter=5000).fit(logit, y).coef_[0][0]


def evaluate(make_model, X, y, splits, make_targets=None, n_splits_per_repeat=5):
    """Cross-validate a model whose TRAINING targets may be changed inside each fold.

    make_model():                     returns a fresh, unfitted model (preprocessing included)
    make_targets(X_train, y_train):   returns the training targets (e.g. smoothed or soft labels);
                                      None = the true 0/1 labels. Always computed from the training
                                      fold only, so the test fold is never used.
    Scoring is always against the TRUE labels of the test fold.

    Returns per-fold ROC-AUC, PR-AUC and Brier, and the calibration slope per repeat
    (each repeat gives every person exactly one out-of-fold prediction).
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    n_repeats = len(splits) // n_splits_per_repeat
    oof = np.full((n_repeats, len(y)), np.nan)
    roc, pr, brier = [], [], []
    for i, (tr, te) in enumerate(splits):
        targets = y[tr] if make_targets is None else make_targets(X.iloc[tr], y[tr])
        p = make_model().fit(X.iloc[tr], targets).predict_proba(X.iloc[te])[:, 1]
        oof[i // n_splits_per_repeat, te] = p
        roc.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
        brier.append(brier_score_loss(y[te], p))
    slopes = [calibration_slope(y, oof[r]) for r in range(n_repeats)]
    return {'roc_auc': np.array(roc), 'pr_auc': np.array(pr), 'brier': np.array(brier),
            'cal_slope': np.array(slopes)}

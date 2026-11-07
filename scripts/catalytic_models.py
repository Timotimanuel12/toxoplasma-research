"""
Custom models built around how Toxoplasma infection works.

Toxoplasma IgG stays positive for life, so being positive at age a means "infected at some point during a years".
The classic epidemiological "catalytic" model says:

    P(positive | x, age) = 1 - exp( - rate(x) * age )

where rate(x) is the person's yearly force of infection (depends on birthplace, poverty, exposures ...).
Taking log-log of both sides gives a linear model with a fixed offset log(age):

    log(-log(1 - p)) = log(rate(x)) + log(age)

Two estimators (scikit-learn compatible):

  CatalyticGLM    log(rate) = b0 + b.x (+ age spline for waning or changing exposure), L2-penalised,
                  complementary log-log link with log(age) offset, fitted by L-BFGS.
  CatalyticBoost  log(rate) = gradient-boosted trees (XGBoost) with a CUSTOM objective = catalytic
                  log-likelihood, so the trees learn the yearly infection rate rather than the probability.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

EPS = 1e-9


def _p_from_eta(eta):
    return -np.expm1(-np.exp(np.clip(eta, -30, 30)))


class CatalyticGLM(BaseEstimator, ClassifierMixin):
    """X must already be numeric/scaled; age (years) is passed through column `age_col` (index)."""

    def __init__(self, age_col=0, alpha=1.0, use_age_in_rate=True):
        self.age_col = age_col
        self.alpha = alpha
        self.use_age_in_rate = use_age_in_rate

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        age = np.clip(X[:, self.age_col], 1, None)
        Z = X if self.use_age_in_rate else np.delete(X, self.age_col, axis=1)
        return Z, np.log(age)

    def fit(self, X, y):
        Z, off = self._split(X)
        y = np.asarray(y, dtype=float)
        n, k = Z.shape
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - self.mu_) / self.sd_

        def loss(w):
            eta = w[0] + Zs @ w[1:] + off
            lam = np.exp(np.clip(eta, -30, 30))
            p = -np.expm1(-lam)
            ll = y * np.log(p + EPS) - (1 - y) * lam                 # log(1-p) = -lam
            # gradient of -ll w.r.t. eta
            g_eta = -(y * lam * np.exp(-lam) / (p + EPS) - (1 - y) * lam)
            grad = np.concatenate([[g_eta.sum()], Zs.T @ g_eta]) / n
            grad[1:] += self.alpha * w[1:] / n
            return -ll.mean() + 0.5 * self.alpha * np.sum(w[1:] ** 2) / n, grad

        w0 = np.zeros(k + 1)
        w0[0] = np.log(max(y.mean(), 1e-3) / np.exp(off).mean())
        self.res_ = minimize(loss, w0, jac=True, method='L-BFGS-B', options={'maxiter': 2000})
        self.w_ = self.res_.x
        self.classes_ = np.array([0, 1])
        return self

    def yearly_rate(self, X):
        Z, _ = self._split(X)
        return np.exp(self.w_[0] + ((Z - self.mu_) / self.sd_) @ self.w_[1:])

    def predict_proba(self, X):
        Z, off = self._split(X)
        p = _p_from_eta(self.w_[0] + ((Z - self.mu_) / self.sd_) @ self.w_[1:] + off)
        return np.column_stack([1 - p, p])


def _catalytic_obj(age_offset):
    def obj(preds, dtrain):
        y = dtrain.get_label()
        eta = np.clip(preds + age_offset, -30, 30)
        lam = np.exp(eta)
        e = np.exp(-lam)
        p = 1 - e
        # d(-ll)/d eta and d2(-ll)/d eta2 for ll = y log p - (1-y) lam
        r = lam * e / np.clip(p, EPS, None)              # d log p / d eta
        grad = -(y * r - (1 - y) * lam)
        # second derivative of log p wrt eta: r * (1 - lam - r)
        hess = -(y * r * (1 - lam - r) - (1 - y) * lam)
        return grad, np.clip(hess, 1e-6, None)
    return obj


class CatalyticBoost(BaseEstimator, ClassifierMixin):
    """XGBoost learning log(yearly infection rate) with the catalytic likelihood. Accepts NaN."""

    def __init__(self, age_col=0, n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8,
                 colsample_bytree=0.7, min_child_weight=10, reg_lambda=5.0, use_age_in_rate=True, random_state=0):
        self.age_col = age_col
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.use_age_in_rate = use_age_in_rate
        self.random_state = random_state

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        off = np.log(np.clip(X[:, self.age_col], 1, None))
        Z = X if self.use_age_in_rate else np.delete(X, self.age_col, axis=1)
        return Z, off

    def fit(self, X, y):
        Z, off = self._prep(X)
        y = np.asarray(y, dtype=float)
        base = np.log(max(y.mean(), 1e-3) / np.exp(off).mean())
        d = xgb.DMatrix(Z, label=y, base_margin=np.full(len(y), base))
        params = dict(max_depth=self.max_depth, eta=self.learning_rate, subsample=self.subsample,
                      colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                      reg_lambda=self.reg_lambda, seed=self.random_state, nthread=1, tree_method='hist')
        self.base_ = base
        self.booster_ = xgb.train(params, d, num_boost_round=self.n_estimators, obj=_catalytic_obj(off))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Z, off = self._prep(X)
        d = xgb.DMatrix(Z, base_margin=np.full(len(Z), self.base_))
        eta = self.booster_.predict(d, output_margin=True) + off
        p = _p_from_eta(eta)
        return np.column_stack([1 - p, p])

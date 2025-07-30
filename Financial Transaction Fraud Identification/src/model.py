from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_unsupervised(X):
    iso = IsolationForest(contamination=0.001, random_state=42)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
    return iso.fit(X), lof.fit_predict(X)

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    return xgb

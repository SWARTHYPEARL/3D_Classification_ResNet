from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)

print(f"X: {clf.predict_proba(X)[:30, 1]}")
print(f"y: {y[:30]}")
print(f"X: {type(clf.predict_proba(X)[:,1])}")
print(f"y: {type(y)}")

print(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

print(roc_auc_score(y, clf.decision_function(X)))
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Dataset import GenDataset
from utils import generate_score


# ─────────────────────────────────────────────
# Dataset — todos os genes de cada classe
# ─────────────────────────────────────────────
data = GenDataset(
    cancer_base_dir="./genbank/cancer",
    normal_base_dir="./genbank/nao_cancer",
)

data.info()

min_classe = int(min(data.Y.sum(), (data.Y == 0).sum()))
#k = min(10, min_classe)
k = 3
print(f"Usando k={k} folds\n")

folds = data.cross_validation_split(k=k)


# ─────────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────────
models = {
    "decisionTree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    ),
    "randomForest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        random_state=42
    ),
    "gradientBoosted": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "svmLinear": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", C=1.0, probability=True, random_state=42))
    ]),
    "svmPolynomial1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="poly", degree=1, C=1.0, gamma="scale", probability=True, random_state=42))
    ]),
    "svmPolynomial2": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="poly", degree=2, C=1.0, gamma="scale", probability=True, random_state=42))
    ]),
    "svmPolynomial3": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="poly", degree=3, C=1.0, gamma="scale", probability=True, random_state=42))
    ]),
    "svmRBF": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ]),
    "svmSigmoid": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="sigmoid", C=1.0, gamma="scale", probability=True, random_state=42))
    ]),
}


# ─────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────
def create_score_dict():
    return {
        "accuracy":    [],
        "f1-score":    [],
        "specificity": [],
        "recall":      [],
        "precision":   [],
        "roc":         [],
    }

scores = {name: create_score_dict() for name in models}

for train_index, test_index in folds:
    X_train, Y_train, X_test, Y_test = data.get_fold(train_index, test_index)

    for name, model in models.items():
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        generate_score(scores[name], Y_test, y_pred, y_prob)


# ─────────────────────────────────────────────
# Resultados
# ─────────────────────────────────────────────
metrics = ["accuracy", "f1-score", "specificity", "recall", "precision", "roc"]

print("\n" + "=" * 65)
print("  RESULTADOS (médias sobre os folds)")
print("=" * 65)

header = f"{'Modelo':<20}" + "".join(f"{m:>12}" for m in metrics)
print(header)
print("-" * len(header))

for model_name, model_scores in scores.items():
    linha = f"{model_name:<20}"
    for metric in metrics:
        media = np.nanmean(model_scores[metric])
        linha += f"{media:>12.4f}"
    print(linha)

print("=" * 65)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
from Dataset import GenDataset
from utils import generate_score


def create_score_dict():
    return {
        "accuracy": [],
        "f1-score": [],
        "specificity": [],
        "recall": [],
        "precision": [],
        "roc": []
    }


data = GenDataset(
    cancer_seq_dir="../Genbank/cancer/BRCA1/sequences",
    normal_seq_dir="../Genbank/nao_cancer/BRCA1/sequences",
)

data.info()

print("Total de amostras:", len(data))
print("Formato de X:", data.X.shape)
print("Formato de Y:", data.Y.shape)

folds = data.cross_validation_split()


svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=42
    ))
])

svm_poly_1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="poly",
        degree=1,
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_poly_2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="poly",
        degree=2,
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_poly_3 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="poly",
        degree=3,
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_sigmoid = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="sigmoid",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])


models = {
    "decisionTree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    "randomForest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),

    "svmLinear": svm_linear,
    "svmPolynomial1": svm_poly_1,
    "svmPolynomial2": svm_poly_2,
    "svmPolynomial3": svm_poly_3,
    "svmRBF": svm_rbf,
    "svmSigmoid": svm_sigmoid,

    "gradientBoosted": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


scores = {}

for name in models.keys():
    scores[name] = create_score_dict()


for fold in folds:
    train_index, test_index = fold

    train_X, train_Y, test_X, test_Y = data.get_fold(train_index, test_index)

    for name, model in models.items():
        model.fit(train_X, train_Y)

        y_pred = model.predict(test_X)

        generate_score(scores[name], test_Y, y_pred)


print("\n===== RESULTADOS =====")

for model_name, model_scores in scores.items():
    print(f"\nModelo: {model_name}")

    for metric_name, values in model_scores.items():
        if metric_name == "roc":
            continue

        print(f"{metric_name}: {np.mean(values):.4f}")
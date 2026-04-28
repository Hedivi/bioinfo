from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
from Dataset import GenDataset
from utils import generate_score

# Cria um dicionário vazio de métricas
def create_score_dict():
    return {
        "accuracy": [],
        "f1-score": [],
        "specificity": [],
        "recall": [],
        "precision": [],
        "roc": []
    }


# Carrega os dados Genbank
data = GenDataset(
    cancer_seq_dir="genbank_output/sequences/cancer",
    normal_seq_dir="genbank_output/sequences/normal",
)
data.info()

print("Total de amostras:", len(data))
print("Formato de X:", data.X.shape)
print("Formato de Y:", data.Y.shape)

# Consegue os índices de cada fold em relação ao dataset carregado
folds = data.cross_validation_split()

# Cria um dicionário de scores para cada modelo
scores = {
    "decisionTree": create_score_dict(),
    "randomForest": create_score_dict(),
    "svm": create_score_dict(),
    "gradientBoosted": create_score_dict()
}

# Define o modelo SVM
# O SVM costuma funcionar melhor com dados normalizados
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True))
])

# Define um dicionário de modelos
models = {
    "decisionTree": DecisionTreeClassifier(),
    "randomForest": RandomForestClassifier(),
    "svm": svm,
    "gradientBoosted": GradientBoostingClassifier()
}

# Para cada conjunto de índices
for fold in folds:

    # Seleciona os índices de teste e treino
    train_index, test_index = fold

    # Seleciona os dados de cada índice
    train_X, train_Y, test_X, test_Y = data.get_fold(train_index, test_index)

    # Para cada modelo
    for name, model in models.items():

        # Treina o modelo
        model.fit(train_X, train_Y)

        # Testa o modelo
        y_pred = model.predict(test_X)

        # Gera as métricas para o modelo testado
        generate_score(scores[name], test_Y, y_pred)


# Imprime os resultados
print("\n===== RESULTADOS =====")

for model_name, model_scores in scores.items():

    print(f"\nModelo: {model_name}")

    for metric_name, values in model_scores.items():

        # A curva ROC não será impressa como média simples
        if metric_name == "roc":
            continue

        print(f"{metric_name}: {np.mean(values):.4f}")

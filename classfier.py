from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Dataset import GenDataset
from utils import generate_score

# Define os diretórios dos dados
cancer_dir = "data/cancer/"
normal_dir = "data/normal/"

# Carrega os dados
data = GenDataset(cancer_dir, normal_dir)

# Consegue os índices de cada fold em relação ao dataset carregado
folds = data.cross_validation_split()

# Define as métricas
score = {"accuracy": [], 
          "f1-score": [],
          "specifity": [],
          "recall": [],
          "precision": [],
          "roc": []}

# Cria um dicionário de scores para cada modelo
scores = {"decisionTree": score.copy(),
          "randomForest": score.copy(),
          "svm": score.copy(),
          "gradientBoosted": score.copy
}

# Define o modelo SVM (Support Vector Machine), ele lida melhor com dados normalizados
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf"))
])

# Define um dicionário de modelos
models = {"decisionTree": DecisionTreeClassifier(),
          "randomForest": RandomForestClassifier(),
          "svm": svm,
          "gradietBoosted": GradientBoostingClassifier()}

# Para cada conjunto de índices 
for fold in folds:
    
    # Seleciona os índices de teste e treino
    train_index, test_index = fold

    # Seleciona os dados de cada índice
    train_X, train_Y, test_X, test_Y = data.get_fold(train_index, test_index)

    # Para cada modelo
    for name, model in models:

        # Treina o modelo
        model.fit(train_X, train_Y)

        # Testa o modelo, y_pred é uma lista com a resposta do modelo
        y_pred = model.predict(test_X)

        # Gera as métricas para o modelo testado
        generate_score(scores[name], test_Y, y_pred)

# Imprime as métricas
print(scores)



    

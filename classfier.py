from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Dataset import GenDataset
from utils import generate_score

cancer_dir = "data/cancer/"
normal_dir = "data/normal/"

data = GenDataset(cancer_dir, normal_dir)
folds = data.cross_validation_split()

scores = {"accuracy": [], 
          "f1-score": [],
          "specifity": [],
          "recall": [],
          "precision": [],
          "roc": []}

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf"))
])

models = {"decisionTree": DecisionTreeClassifier(),
          "randomForest": RandomForestClassifier(),
          "svm": svm,
          "gradietBoosted": GradientBoostingClassifier()}

for fold in folds:
    train_index, test_index = fold
    train_X, train_Y, test_X, test_Y = data.get_fold(train_index, test_index)

    for model in models:

        model.fit(train_X, train_Y)
        y_pred = model.predict(test_X)

        generate_score(scores, test_Y, y_pred)

print(scores)



    

import pandas as pd
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Base Learners
base_models = [
    ('dt', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
    ('rf', RandomForestClassifier())
]

# Meta Learner
meta_model = LogisticRegression()

# Stacking Classifier
model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))

# Save Model
pickle.dump(
    model,
    open("models/stacking_classifier.pkl", "wb")
)

print("Model Saved")

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model1 in models.items():

    model1.fit(X_train, y_train)

    pred = model1.predict(X_test)

    print(
        name,
        accuracy_score(y_test, pred)
    )

print(
    "Stacking",
    accuracy_score(y_test, y_pred)
)
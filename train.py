
import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split data before fitting the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Build a preprocessing + classification pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 4. Train only on training data
model.fit(X_train, y_train)

# 5. Evaluate on unseen test data
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)

# 6. Save the complete fitted pipeline
joblib.dump(
    {
        "model": model,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names
    },
    "iris_model.joblib"
)

print("Model saved successfully!")

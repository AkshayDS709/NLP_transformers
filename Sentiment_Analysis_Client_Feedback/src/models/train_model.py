import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load embeddings
df = pd.read_csv("data/sentiment_embeddings.csv")
X = df.drop(columns=["sentiment"])
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True),
    "LogisticRegression": LogisticRegression()
}

# Train and evaluate models
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1 Score: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    joblib.dump(model, f"src/models/{name}_model.pkl")

joblib.dump(best_model, "src/models/best_model.pkl")
print("Best model saved successfully.")

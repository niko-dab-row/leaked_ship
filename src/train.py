import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing import preprocess


train_df = pd.read_csv("../data/train.csv")

train_df = preprocess(train_df)

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

accuracy = accuracy_score(
    y_valid,
    predictions
)

print(f"Validation accuracy: {accuracy:.4f}")

joblib.dump(
    model,
    "../models/random_forest.pkl"
)

print("Model saved.")
import joblib
import pandas as pd

from preprocessing import preprocess


model = joblib.load(
    "../models/random_forest.pkl"
)

test_df = pd.read_csv(
    "../data/test.csv"
)

passenger_ids = test_df["PassengerId"]

test_df = preprocess(test_df)

predictions = model.predict(test_df)

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("Submission file created.")
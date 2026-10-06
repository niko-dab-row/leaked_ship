import pandas as pd


def preprocess(df):

    df = df.copy()

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract title from name
    df["Title"] = (
        df["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
    )

    common_titles = [
        "Mr", "Mrs", "Miss", "Master"
    ]

    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles),
        "Other"
    )

    # Encode categoricals
    df["Sex"] = df["Sex"].map(
        {"male": 0, "female": 1}
    )

    df = pd.get_dummies(
        df,
        columns=["Embarked", "Title"],
        drop_first=True
    )

    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df
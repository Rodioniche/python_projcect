import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
categorical_cols = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "poutcome"]
required_columns = numeric_cols + categorical_cols

model = None
scaler = None
encoder = None
is_model_trained = False


def train_model(data_path, test_size=0.2, random_state=42):
    global model, scaler, encoder, is_model_trained

    try:
        df = pd.read_csv(data_path, sep=";")

        y = df['y'].map({"yes": 1, "no": 0})
        X = df[required_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        encoder = OneHotEncoder(drop="first", sparse_output=False)
        encoder.fit(X_train[categorical_cols])

        X_train_processed = _transform_features(X_train)
        X_test_processed = _transform_features(X_test)

        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=0.1,
            solver='liblinear'
        )
        model.fit(X_train_processed, y_train)

        is_model_trained = True
        return True, "Model trained successfully!"
    except Exception as e:
        return False, f"Training error: {str(e)}"


def _transform_features(X):
    X_num = scaler.transform(X[numeric_cols])
    X_cat = encoder.transform(X[categorical_cols])
    return np.hstack([X_num, X_cat])


def is_trained():
    return is_model_trained


is_trained()
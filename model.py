import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)

numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
categorical_cols = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "poutcome"]
required_columns = numeric_cols + categorical_cols

model = None
scaler = None
encoder = None
is_model_trained = False
best_threshold = 0.5
metrics = {}


def train_model(data_path, test_size=0.2, random_state=42):
    global model, scaler, encoder, metrics, best_threshold, is_model_trained

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

        _evaluate_model(X_test_processed, y_test)

        is_model_trained = True
        return True, "Model trained successfully!"
    except Exception as e:
        return False, f"Training error: {str(e)}"


def _transform_features(X):
    X_num = scaler.transform(X[numeric_cols])
    X_cat = encoder.transform(X[categorical_cols])
    return np.hstack([X_num, X_cat])


def _evaluate_model(X, y):
    global metrics, best_threshold

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_f1 = 0
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        f1 = f1_score(y, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'classification_report': classification_report(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }


def is_trained():
    return is_model_trained
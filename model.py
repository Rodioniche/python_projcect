import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
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
metrics = {}
best_threshold = 0.5


def train_model(data_path, test_size=0.2, random_state=42):
    global model, scaler, encoder, is_model_trained, metrics, best_threshold

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

        evaluate_model(X_test_processed, y_test)
        
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


def evaluate_model(X, y):
    global metrics, best_threshold
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    from sklearn.metrics import f1_score
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

def get_model_metrics():
    global is_model_trained, metrics
    
    if not is_model_trained:
        return False, "Model not trained", None
    
    return True, "Metrics retrieved", {
        'accuracy': metrics.get('accuracy'),
        'roc_auc': metrics.get('roc_auc'),
        'classification_report': metrics.get('classification_report'),
        'confusion_matrix': metrics.get('confusion_matrix')
    }

def save_model(path='model_artifacts.joblib'):
    global model, scaler, encoder, metrics
    
    if model is None:
        return False, "Model not trained"
    
    try:
        artifacts = {
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'metrics': metrics
        }
        joblib.dump(artifacts, path)
        return True, f"Model saved to {path}"
    except Exception as e:
        return False, f"Save error: {str(e)}"

def load_model(model_path):
    global model, scaler, encoder, metrics, is_model_trained, numeric_cols, categorical_cols
    
    try:
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        scaler = artifacts['scaler']
        encoder = artifacts['encoder']
        numeric_cols = artifacts.get('numeric_cols', numeric_cols)
        categorical_cols = artifacts.get('categorical_cols', categorical_cols)
        metrics = artifacts.get('metrics', {})
        is_model_trained = True
        return True, "Model loaded successfully!"
    except Exception as e:
        return False, f"Load error: {str(e)}"
    
def predict(input_data):
    global model, scaler, encoder, is_model_trained
    
    if not is_model_trained:
        return False, "Model is not trained. Please train or load a model first.", None
    
    try:
        if isinstance(input_data, str):
            df = pd.read_csv(input_data, sep=";")
        else:
            df = input_data.copy()
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}", None
        
        processed_data = _transform_features(df[required_columns])
        
        probabilities = model.predict_proba(processed_data)[:, 1]
        df['predicted_probability'] = probabilities
        
        return True, "Prediction successful", df
    except Exception as e:
        return False, f"Prediction error: {str(e)}", None
    
def generate_analysis_plot(data, feature):
    if feature not in data.columns:
        return None
        
    plt.figure(figsize=(12, 8), dpi=100)
    
    if data[feature].dtype in ['int64', 'float64']:
        if data[feature].nunique() > 20:
            bins = np.linspace(data[feature].min(), data[feature].max(), 11)
            data['binned'] = pd.cut(data[feature], bins=bins)
            sns.boxplot(
                x='binned', 
                y='predicted_probability', 
                data=data,
                showfliers=False
            )
            plt.xticks(rotation=45)
            plt.xlabel(f"{feature} (binned)")
        else:
            sns.boxplot(
                x=feature, 
                y='predicted_probability', 
                data=data,
                showfliers=False
            )
            plt.xlabel(feature)
        
        plt.title(f'Prediction Distribution by {feature}')
        plt.ylabel('Predicted Probability')
    
    else:
        value_counts = data[feature].value_counts()
        top_categories = value_counts.head(10).index
        filtered_data = data[data[feature].isin(top_categories)]
        
        sns.boxplot(
            x=feature, 
            y='predicted_probability', 
            data=filtered_data,
            showfliers=False
        )
        plt.title(f'Prediction Distribution by {feature}')
        plt.ylabel('Predicted Probability')
        plt.xlabel(feature)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    return buf

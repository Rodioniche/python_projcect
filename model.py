import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, f1_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

model = None
scaler = None
encoder = None
best_threshold = 0.5
metrics = {}
is_model_trained = False

base_numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
base_categorical_cols = ["job", "marital", "education", "default", 
                         "housing", "loan", "contact", "month", "poutcome"]
required_columns = base_numeric_cols + base_categorical_cols

numeric_cols = base_numeric_cols.copy()
categorical_cols = base_categorical_cols.copy()

def create_features(df):
    df = df.copy()

    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['balance_duration_ratio'] = df['balance'] / (df['duration'] + 1)
    df['contact_rate'] = df['previous'] / (df['campaign'] + 1)

    df['job'] = df['job'].replace(['unknown', 'other'], 'other_job')
    df['education'] = df['education'].replace(['unknown'], 'other_education')
    
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                            labels=['young', 'adult', 'senior', 'elderly'])

    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)

    df['has_loan'] = (df['housing'] == 'yes') | (df['loan'] == 'yes')
    df['product_engagement'] = df['campaign'] * df['previous']
    
    return df

def train_model(data_path, test_size=0.2, random_state=42):
    global model, scaler, encoder, metrics, is_model_trained, numeric_cols, categorical_cols
    
    try:
        df = pd.read_csv(data_path, sep=";")
        y = df['y'].map({"yes": 1, "no": 0})
        df = create_features(df)  
        
        numeric_cols = base_numeric_cols + ['balance_per_age', 'balance_duration_ratio', 
                                          'month_num', 'contact_rate', 'product_engagement']
        categorical_cols = base_categorical_cols + ['age_group', 'has_loan']
        
        missing_num = [col for col in numeric_cols if col not in df.columns]
        missing_cat = [col for col in categorical_cols if col not in df.columns]
        
        if missing_num or missing_cat:
            return False, f"Missing features: {missing_num + missing_cat}", None
            
        X = df[numeric_cols + categorical_cols]

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

        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

        scale_pos_weight = (len(y_train_balanced) - sum(y_train_balanced)) / sum(y_train_balanced)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='auc'
        )
 
        model.fit(X_train_balanced, y_train_balanced)

        _evaluate_model(X_test_processed, y_test)
        
        is_model_trained = True
        return True, "Model trained successfully with XGBoost!"
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
    roc_auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': best_f1,
        'classification_report': classification_report(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'best_threshold': best_threshold
    }
    
    print(f"Best F1: {best_f1:.4f} at threshold: {best_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

def save_model(path='model_artifacts.joblib'):
    global model, scaler, encoder, metrics, numeric_cols, categorical_cols
    
    if model is None:
        return False, "Model not trained"
    
    try:
        artifacts = {
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'numeric_cols': numeric_cols, 
            'categorical_cols': categorical_cols,
            'metrics': metrics,
            'best_threshold': best_threshold
        }
        joblib.dump(artifacts, path)
        return True, f"Model saved to {path}"
    except Exception as e:
        return False, f"Save error: {str(e)}"

def load_model(model_path):
    global model, scaler, encoder, metrics, is_model_trained, numeric_cols, categorical_cols, best_threshold
    
    try:
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        scaler = artifacts['scaler']
        encoder = artifacts['encoder']
        numeric_cols = artifacts['numeric_cols']
        categorical_cols = artifacts['categorical_cols']
        metrics = artifacts.get('metrics', {})
        best_threshold = artifacts.get('best_threshold', 0.5)
        is_model_trained = True
        return True, "Model loaded successfully!"
    except Exception as e:
        return False, f"Load error: {str(e)}"

def predict(input_data):
    global model, scaler, encoder, is_model_trained, best_threshold, numeric_cols, categorical_cols
    if not is_model_trained:
        return False, "Model not trained", None
    
    try:
        if isinstance(input_data, str):
            df = pd.read_csv(input_data, sep=";")
        else:
            df = input_data.copy()
        
        df = create_features(df)
        
        required = numeric_cols + categorical_cols
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}", None
        
        X_processed = np.hstack([
            scaler.transform(df[numeric_cols]),
            encoder.transform(df[categorical_cols])
        ])

        y_proba = model.predict_proba(X_processed)[:, 1]
        df['predicted_probability'] = y_proba
        df['prediction'] = (y_proba >= best_threshold).astype(int)
        
        return True, "Prediction successful", df
    except Exception as e:
        return False, f"Prediction error: {str(e)}", None

def get_model_metrics():
    global is_model_trained, metrics
    
    if not is_model_trained:
        return False, "Model not trained", None
    
    return True, "Metrics retrieved", {
        'accuracy': metrics.get('accuracy'),
        'roc_auc': metrics.get('roc_auc'),
        'classification_report': metrics.get('classification_report'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'best_threshold': metrics.get('best_threshold', 0.5)
    }

def is_trained():
    return is_model_trained

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

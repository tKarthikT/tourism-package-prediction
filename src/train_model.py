import os
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

ARTIFACT_DIR = 'artifacts'
MODEL_DIR = 'models'

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train = pd.read_csv(f'{ARTIFACT_DIR}/X_train.csv')
    X_test = pd.read_csv(f'{ARTIFACT_DIR}/X_test.csv')
    y_train = pd.read_csv(f'{ARTIFACT_DIR}/y_train.csv').squeeze('columns')
    y_test = pd.read_csv(f'{ARTIFACT_DIR}/y_test.csv').squeeze('columns')
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numeric_features), ('cat', categorical_pipeline, categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))])
    param_grid = {'model__n_estimators': [200, 300], 'model__max_depth': [None, 10, 20], 'model__min_samples_split': [2, 5], 'model__min_samples_leaf': [1, 2], 'model__max_features': ['sqrt', 'log2']}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        'best_cv_f1_score': float(grid_search.best_score_),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred)),
        'test_recall': float(recall_score(y_test, y_pred)),
        'test_f1_score': float(f1_score(y_test, y_pred)),
        'test_roc_auc': float(roc_auc_score(y_test, y_prob)),
        'best_parameters': grid_search.best_params_
    }
    joblib.dump(best_model, f'{MODEL_DIR}/best_model.pkl')
    with open(f'{MODEL_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print('Best Parameters:', grid_search.best_params_)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_model()

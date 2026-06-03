import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

from data_processing import build_full_pipeline, assign_high_risk_label

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

def main():
    df = pd.read_csv('data.csv', parse_dates=['TransactionStartTime'])
    pipeline = build_full_pipeline()
    customer_df = pipeline.named_steps['aggregator'].fit_transform(df)
    y_df = assign_high_risk_label(customer_df)
    X = pipeline.fit_transform(df)
    y = y_df.set_index('CustomerId').loc[customer_df['CustomerId']]['is_high_risk'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_params = {'C': [0.1, 1, 10]}
    lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=3, scoring='roc_auc')
    lr.fit(X_train, y_train)
    
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
    rf.fit(X_train, y_train)
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CreditRiskModel")
    
    best_model = None
    best_score = 0.0
    for model, name in [(lr, "LogisticRegression"), (rf, "RandomForest")]:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.best_params_)
            metrics = evaluate(model.best_estimator_, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model.best_estimator_, name)
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model.best_estimator_
    
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved with ROC-AUC: {best_score:.4f}")

if __name__ == "__main__":
    main()

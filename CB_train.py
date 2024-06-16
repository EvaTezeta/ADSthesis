import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import catboost as cb
import joblib
import numpy as np
import random
import json

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load dataset
df = pd.read_csv('C:/Users/j.perdeck/BC_Prediction/Eva/CB/4_PredictedData_Students.csv')

# Drop unnecessary columns
data = df.drop(['Probability', 'Prediction', 'PAT_ENC_CSN_ID'], axis=1)

# Prepare your data
X = data.drop(["outcome"], axis=1)  # Features
y = data["outcome"]  # Target variable

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "class_weights": [1, trial.suggest_float("class_weight_1", 1, 12)]
    }

    model = cb.CatBoostClassifier(**params, silent=True, random_seed=SEED)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    
    # Get predicted probabilities for validation set
    probabilities = model.predict_proba(X_val)[:, 1]
    
    # Calculate recall on validation set
    recall = recall_score(y_val, probabilities > 0.5)
    
    return recall  # Maximize recall

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30)

# Best hyperparameters
best_params = study.best_params
print('Best hyperparameters:', best_params)

# Save the best hyperparameters to a text file
with open('best_hyperparameters_cb.txt', 'w') as file:
    file.write(json.dumps(best_params, indent=4))

# Train CatBoost with best hyperparameters
best_params["class_weights"] = [1, best_params.pop("class_weight_1")]
model_cb = cb.CatBoostClassifier(**best_params, silent=True, random_seed=SEED)
model_cb.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_cb, 'model_cb.pkl')

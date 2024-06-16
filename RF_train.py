import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import optuna
import joblib
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
    # Define hyperparameter space
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    class_weight_option = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample', 'custom'])

    # Handle custom class weight
    if class_weight_option == 'custom':
        class_weight = {0: 1, 1: 11}
    else:
        class_weight = class_weight_option

    max_samples = trial.suggest_float('max_samples', 0.5, 1.0) if bootstrap else None

    # Create a Random Forest Classifier with suggested hyperparameters
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight=class_weight,
        max_samples=max_samples,
        random_state=SEED
    )

    # Train the model
    rf_clf.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred_val = rf_clf.predict(X_val)

    # Calculate recall on the validation set
    recall = recall_score(y_val, y_pred_val, average='macro')
    
    return recall

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)

# Save the best hyperparameters to a text file
with open('best_hyperparameters_rf.txt', 'w') as file:
    file.write(json.dumps(best_params, indent=4))

# Handle custom class weight for the best parameters
if best_params['class_weight'] == 'custom':
    best_params['class_weight'] = {0: 1, 1: 11}

# Train the final model with the best hyperparameters
model_rf = RandomForestClassifier(
    **best_params,
    random_state=SEED
)
model_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_rf, 'model_rf.joblib')

print("Model training complete and saved as 'model_rf.joblib'")

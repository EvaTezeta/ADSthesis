import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import numpy as np
import random

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

# Load the trained model
model_cb = joblib.load('model_cb.pkl')

# Make final predictions on the test set
y_pred_proba = model_cb.predict_proba(X_test)[:, 1]

# Define the threshold
threshold = 0.4

# Apply threshold to convert probabilities to binary predictions
y_pred_binary = (y_pred_proba > threshold).astype(int)

# Accuracy, Sensitivity (Recall), Precision, F1 Score, and Brier Score for test set
accuracy_test = accuracy_score(y_test, y_pred_binary)
recall_test = recall_score(y_test, y_pred_binary)  # Default is binary recall
precision_test = precision_score(y_test, y_pred_binary)  # Default is binary precision
f1_test = f1_score(y_test, y_pred_binary)  # Default is binary F1 score
brier_score_test = brier_score_loss(y_test, y_pred_proba)

# Calculate specificity
tn, fp, fn, _ = confusion_matrix(y_test, y_pred_binary).ravel()
specificity_test = tn / (tn + fp)

# Calculate percentages of false negatives and true negatives
false_negative_percentage = (fn / len(y_test)) * 100
true_negative_percentage = (tn / len(y_test)) * 100

# ROC AUC
fpr_test, tpr_test, _ = roc_curve(y_test,y_pred_proba)
roc_auc_test = auc(fpr_test, tpr_test)

# Precision-Recall AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc_test = auc(recall_curve, precision_curve)

print('CatBoost Evaluation:')
print('Accuracy:', round(accuracy_test, 3))
print('Sensitivity:', round(recall_test, 3))
print('Specificity:', round(specificity_test, 3))
print('Precision:', round(precision_test, 3))
print('F1 Score:', round(f1_test, 3))
print('Brier Score:', round(brier_score_test, 3))
print('ROC AUC:', round(roc_auc_test, 3))
print('PR AUC:', round(pr_auc_test, 3))

# Print percentages of false negatives and true negatives
print('False Negative Percentage:', round(false_negative_percentage, 2), '%')
print('True Negative Percentage:', round(true_negative_percentage, 2), '%')

# Confusion Matrix for test set
cm = confusion_matrix(y_test,y_pred_binary)

# Define class labels
class_labels = ['Negative', 'Positive']

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('CatBoost Threshold = 0.4')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.savefig("plot_cm_cb_4.png")
plt.show()

# ROC Curve for test set
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve for CatBoost')
plt.legend(loc="lower right")
plt.savefig("plot_auc_cb.png")
plt.show()

# Precision-Recall Curve for test set
plt.figure()
plt.plot(recall_curve, precision_curve, color="blue", lw=2, label="Precision-Recall curve (area = %0.2f)" % pr_auc_test)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for CatBoost")
plt.legend(loc="lower left")
plt.savefig("plot_prc_cb.png")
plt.show()

# Histogram of predicted probabilities
# Separate positive and negative outcomes
positive_probs = y_pred_proba[y_test == 1]
negative_probs = y_pred_proba[y_test == 0]

plt.figure()
# Plot histograms
plt.hist(negative_probs, bins=100, alpha=0.5, color='grey', label='Negative', density=False)
plt.hist(positive_probs, bins=100, alpha=0.5, color='orange', label='Positive', density=False)

# Add a vertical line for the threshold
plt.axvline(threshold, color='blue', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(title='Outcome')

# Add threshold text
plt.text(threshold, plt.ylim()[1]*0.9, '  threshold', horizontalalignment='left', verticalalignment='center', color='blue', fontsize=12, fontweight='bold')

# Show the plot
plt.title('Distribution of Predicted Probabilities for CatBoost')
plt.savefig("plot_histogram_cb.png")
plt.show()


# SHAP Plot
plt.figure()
explainer = shap.TreeExplainer(model_cb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("CatBoost Feature Importance")
plt.xlabel('Feature value impact on model output')
plt.annotate('Lower risk', xy=(0.05, -0.1), xycoords='axes fraction', ha='center', va='center',
             arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate('Higher risk', xy=(0.95, -0.1), xycoords='axes fraction', ha='center', va='center',
             arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1, headwidth=5))
plt.savefig("plot_shap_cb_test.png")
plt.show()

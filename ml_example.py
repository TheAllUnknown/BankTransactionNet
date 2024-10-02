import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Generate a random classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define a pipeline with StandardScaler, SelectKBest for feature selection, and SVM for classification
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif)),
    ('svm', SVC(probability=True))  # Enable probability for AUC calculation
])

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'select__k': [5, 10, 15],  # Number of top features to select
    'svm__C': np.logspace(-3, 3, 10),  # Regularization parameter
    'svm__gamma': ['scale', 'auto'],  # Kernel coefficient
    'svm__kernel': ['linear', 'rbf', 'poly']  # Kernel types
}

# Set up cross-validation and randomized search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=20, scoring='roc_auc', cv=cv, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters from RandomizedSearchCV
print("Best parameters found: ", random_search.best_params_)

# Predict on the test set
y_pred = random_search.predict(X_test)
y_pred_prob = random_search.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

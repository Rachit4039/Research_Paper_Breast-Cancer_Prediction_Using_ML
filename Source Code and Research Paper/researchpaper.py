# #  Breast Cancer Classification Using Machine Learning

# ##  Research Overview
# This project focuses on the classification of breast cancer tumors as **malignant (0)** or **benign (1)** using various machine learning algorithms.  
# We use the built-in dataset from scikit-learn and apply multiple models to compare their performance.

# ##  Objectives
# - To preprocess and analyze the dataset  
# - To apply different machine learning models  
# - To evaluate models using metrics like Accuracy, Precision, Recall, and F1-score  
# - To identify the best performing model  

# ##  Models Used
# - Logistic Regression  
# - Decision Tree  

# - Random Forest  
# - Support Vector Machine (SVM)  

# ##  Dataset
# The dataset used is the **Breast Cancer Wisconsin Dataset**, available in the scikit-learn library.

# ##  Team Members
# - Rachit Joshi(2210992111)
# - Sukriti (2210992417)
# - Parul (2210992033)


# ##  Course Details
# - Subject: COOP-II (22CS421)  
# - Department: Computer Science Engineering  
# - Batch: 2022  

# ## 📅 Academic Year
# 2025–2026



# ================================
# 1. IMPORT LIBRARIES
# ================================

import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Dataset
from sklearn.datasets import load_breast_cancer

# Preprocessing & splitting
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve, auc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 2. LOAD DATASET
# ================================

# Load built-in breast cancer dataset
data = load_breast_cancer()

# Features (independent variables)
X = data.data

# Target (dependent variable: 0 = malignant, 1 = benign)
y = data.target

# Convert to DataFrame for easier understanding
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Display first few rows
print(df.head())


# ================================
# 3. TRAIN-TEST SPLIT
# ================================

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 4. FEATURE SCALING
# ================================

# Standardize features (important for models like SVM)
scaler = StandardScaler()

# Fit scaler on training data and transform both sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================================
# 5. DEFINE MODELS
# ================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}


# ================================
# 6. TRAIN & EVALUATE MODELS
# ================================

results = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results.append([name, acc, prec, rec, f1])


# Convert results into DataFrame
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print("\nFinal Results:")
print(results_df)


# ================================
# 7. DATA VISUALIZATION
# ================================

# Class distribution
sns.countplot(x=y)
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# ================================
# 8. CONFUSION MATRICES
# ================================

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()


# ================================
# 9. ACCURACY COMPARISON
# ================================

model_names = results_df["Model"]
accuracies = results_df["Accuracy"]

plt.figure()
plt.bar(model_names, accuracies)
plt.title("Model Accuracy")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


# ================================
# 10. PRECISION / RECALL / F1
# ================================

results_df.set_index("Model")[["Precision", "Recall", "F1 Score"]].plot(kind='bar')

plt.title("Precision, Recall, F1 Score Comparison")
plt.ylabel("Score")
plt.show()


# ================================
# 11. FEATURE IMPORTANCE (RANDOM FOREST)
# ================================

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Get importance scores
importances = rf_model.feature_importances_
feature_names = data.feature_names

# Create DataFrame
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feat_df.head())


# ================================
# 12. ROC CURVE (SVM)
# ================================

# Train SVM again
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Get probabilities
y_prob = svm_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SVM)")
plt.legend()
plt.show()


# ================================
# 13. CROSS-VALIDATION
# ================================

# Perform 5-fold cross-validation on SVM
scores = cross_val_score(svm_model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
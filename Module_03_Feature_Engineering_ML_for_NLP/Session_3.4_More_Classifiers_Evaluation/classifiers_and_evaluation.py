# Module_03_Feature_Engineering_ML_for_NLP/Session_3.4_More_Classifiers_Evaluation/classifiers_and_evaluation.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # Support Vector Classifier for linear SVMs

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd # For displaying confusion matrix nicely
import matplotlib.pyplot as plt # For plotting confusion matrix (optional)
import seaborn as sns # For plotting confusion matrix (optional)

# --- 1. Sample Data (same as previous session for consistency) ---
binary_documents = [
    "This is a fantastic movie, loved it!",               # Positive (1)
    "Absolutely wonderful experience, highly recommended.", # Positive (1)
    "The plot was engaging and the actors were brilliant.",# Positive (1)
    "I really enjoyed this film.",                         # Positive (1)
    "What a terrible waste of time.",                      # Negative (0)
    "I hated every moment of it.",                         # Negative (0)
    "The storyline was boring and predictable.",           # Negative (0)
    "Would not recommend this to anyone."                  # Negative (0)
]
binary_labels = [1, 1, 1, 1, 0, 0, 0, 0]      # 1 for Positive, 0 for Negative

print("--- Sample Binary Classification Data Loaded ---")

# --- 2. Feature Extraction (TF-IDF) ---
vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(binary_documents)
y_labels = np.array(binary_labels)

# --- 3. Splitting Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_labels,
    test_size=0.25, # 2 out of 8 samples for testing
    random_state=42
)
print(f"Test set size: {X_test.shape[0]} samples.")
print(f"Actual labels for test set (y_test): {y_test}\n")

# --- 4. Model Training and Evaluation ---
# We'll define a list of models to train and evaluate
models_to_evaluate = [
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=42)), # liblinear is good for small datasets
    ("Linear SVM (LinearSVC)", LinearSVC(random_state=42, dual='auto')) # dual='auto' for newer scikit-learn
]

for model_name, model in models_to_evaluate:
    print(f"--- Evaluating: {model_name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print(f"Predicted labels: {y_pred}")

    # --- Evaluation Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    # For binary classification, specify pos_label for precision, recall, f1 if needed,
    # or use default which assumes the positive class is 1.
    # `average='binary'` is default for binary classification if y_true and y_pred are binary.
    precision = precision_score(y_test, y_pred, zero_division=0) # zero_division to handle cases with no P or TP
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nMetrics for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix (y_true vs y_pred):\n{cm}")
    # For better display (optional, requires pandas)
    try:
        # Ensure labels are consistent for CM display if not just 0,1
        class_names = ["Negative (0)", "Positive (1)"]
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(f"  Confusion Matrix (DataFrame):\n{df_cm}")

        # Plotting the confusion matrix (optional, requires matplotlib & seaborn)
        # plt.figure(figsize=(6,4))
        # sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        # plt.title(f'Confusion Matrix: {model_name}')
        # plt.ylabel('Actual Label')
        # plt.xlabel('Predicted Label')
        # plt.show() # This would pop up a plot
        # print("   (Plotting commented out for non-interactive execution in this demo script)")
    except ImportError:
        print("   (Pandas/Matplotlib/Seaborn not installed, skipping enhanced CM display/plot)")
    except Exception as e_plot:
        print(f"   (Error during enhanced CM display/plot: {e_plot})")


    # --- Classification Report ---
    # Provides precision, recall, F1-score per class, and averages
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(f"\n  Classification Report:\n{report}")
    print("-" * 50 + "\n")


print("--- Overall Notes on This Small Dataset ---")
print("1. With only 2 test samples, metrics can be extreme (0 or 1).")
print("2. 'zero_division=0' in precision/recall/f1/classification_report handles cases where a")
print("   denominator is zero (e.g., no true positives AND no false positives for a class).")
print("3. Logistic Regression and LinearSVC often perform very well on text data.")
print("4. The choice of `solver` for LogisticRegression or `dual` for LinearSVC can depend on dataset size and sklearn version.")

# Example of multiclass classification report structure (conceptual, not run with current binary data)
# if False: # Placeholder for a multiclass example
#     y_true_multiclass = [0, 1, 2, 0, 1, 2]
#     y_pred_multiclass = [0, 1, 1, 0, 2, 2]
#     target_names_multiclass = ['class_A', 'class_B', 'class_C']
#     report_multiclass = classification_report(y_true_multiclass, y_pred_multiclass, target_names=target_names_multiclass)
#     print("\nExample Multiclass Classification Report Structure:")
#     print(report_multiclass)
#     # It would show precision, recall, f1 for class_A, class_B, class_C,
#     # plus macro average and weighted average.

# Session 3.4: More Classifiers and Evaluation Metrics

Welcome to Session 3.4! In the previous session, we trained our first text classifier (Multinomial Naive Bayes) and made some basic predictions. Now, we'll explore other common classifiers suitable for text data and, critically, learn how to formally **evaluate the performance** of our classification models using various metrics and tools like the confusion matrix.

## Learning Objectives:

*   Get introduced to other common machine learning classifiers for text:
    *   **Logistic Regression**
    *   **Support Vector Machines (SVMs)** (specifically Linear SVMs)
*   Understand and interpret key evaluation metrics for classification tasks:
    *   **Accuracy** (and its limitations)
    *   **Precision**
    *   **Recall (Sensitivity)**
    *   **F1-Score**
*   Learn how to interpret a **Confusion Matrix**.
*   Use Scikit-learn's `classification_report` and `confusion_matrix` functions for comprehensive evaluation.

## More Classifiers for Text:

While Multinomial Naive Bayes is a good baseline, other algorithms often provide better performance, especially with larger datasets.

1.  **Logistic Regression:**
    *   **Concept:** Despite its name, Logistic Regression is a linear model used for **binary classification** problems (it can be extended for multiclass classification, e.g., using One-vs-Rest). It models the probability of a binary outcome using a logistic function (sigmoid function).
    *   **For Text:** It works well with high-dimensional sparse data like TF-IDF features. It tries to find a linear decision boundary that separates the classes.
    *   **Scikit-learn:** `sklearn.linear_model.LogisticRegression`
    *   **Pros:** Interpretable (coefficients can indicate feature importance), efficient, performs well.
    *   **Cons:** Assumes a linear relationship between features and the log-odds of the outcome.

2.  **Support Vector Machines (SVMs):**
    *   **Concept:** SVMs are powerful supervised learning models that can be used for classification and regression. For classification, an SVM tries to find a hyperplane in an N-dimensional space (where N is the number of features) that best separates the data points of different classes. The "support vectors" are the data points closest to the hyperplane.
    *   **Linear SVMs:** For text data (which is often high-dimensional), Linear SVMs (using a linear kernel) are particularly effective and computationally efficient.
    *   **Scikit-learn:** `sklearn.svm.SVC` (with `kernel='linear'`) or the more optimized `sklearn.svm.LinearSVC`. `LinearSVC` is generally preferred for text due to its scalability.
    *   **Pros:** Very effective in high-dimensional spaces (good for text), robust to overfitting in high dimensions if the margin is large.
    *   **Cons:** Can be computationally intensive to train with very large datasets (though `LinearSVC` is better), less directly interpretable than Logistic Regression.

## Evaluation Metrics for Classification:

Simply looking at raw predictions isn't enough. We need quantitative metrics to assess how well our model is performing, especially on the unseen test data. These metrics are usually derived from the **Confusion Matrix**.

**Assume a binary classification problem (e.g., Positive vs. Negative):**
*   **True Positives (TP):** Instances correctly predicted as Positive.
*   **True Negatives (TN):** Instances correctly predicted as Negative.
*   **False Positives (FP) (Type I Error):** Instances incorrectly predicted as Positive (were actually Negative).
*   **False Negatives (FN) (Type II Error):** Instances incorrectly predicted as Negative (were actually Positive).

1.  **Confusion Matrix:**
    *   A table that summarizes the performance of a classification algorithm.
    *   Rows typically represent the actual classes, and columns represent the predicted classes (or vice-versa).
    *   Example for binary classification:
        ```
                        Predicted Negative   Predicted Positive
        Actual Negative         TN                 FP
        Actual Positive         FN                 TP
        ```
    *   It helps visualize where the model is making errors.

2.  **Accuracy:**
    *   **Formula:** `(TP + TN) / (TP + TN + FP + FN)` (Total correct predictions / Total predictions)
    *   **Interpretation:** The proportion of total predictions that were correct.
    *   **Limitation:** Can be misleading for **imbalanced datasets**. If one class is much more frequent than others, a model that always predicts the majority class can achieve high accuracy but be useless for predicting the minority class.
        *   Example: In spam detection, if 95% of emails are "ham" and 5% are "spam", a model that predicts all emails as "ham" has 95% accuracy but fails to identify any spam.

3.  **Precision:**
    *   **Formula:** `TP / (TP + FP)`
    *   **Interpretation:** Of all the instances the model predicted as Positive, what proportion were *actually* Positive?
    *   **Focus:** Minimizing False Positives. High precision means the model is trustworthy when it predicts Positive.
    *   **Use Case:** Important when the cost of a False Positive is high (e.g., in medical diagnosis, incorrectly diagnosing a healthy patient as sick; or in spam filtering, incorrectly marking a legitimate email as spam).

4.  **Recall (Sensitivity or True Positive Rate):**
    *   **Formula:** `TP / (TP + FN)`
    *   **Interpretation:** Of all the instances that were *actually* Positive, what proportion did the model *correctly* identify as Positive?
    *   **Focus:** Minimizing False Negatives. High recall means the model finds most of the actual Positive instances.
    *   **Use Case:** Important when the cost of a False Negative is high (e.g., in medical diagnosis, failing to diagnose a sick patient; or in fraud detection, failing to detect a fraudulent transaction).

5.  **F1-Score:**
    *   **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
    *   **Interpretation:** The harmonic mean of Precision and Recall. It provides a single score that balances both concerns.
    *   **Range:** 0 to 1 (higher is better).
    *   **Use Case:** Useful when you need a balance between Precision and Recall, especially if the class distribution is uneven.

**For Multiclass Classification:**
These metrics can be extended to multiclass problems, often by calculating them "per class" (e.g., precision for Class A, recall for Class A) and then averaging them:
*   **Macro Average:** Calculate the metric independently for each class and then take the unweighted average. Treats all classes equally.
*   **Weighted Average:** Calculate the metric for each class and then average, weighted by the number of true instances for each class (support). Accounts for class imbalance.
*   **Micro Average:** Aggregate the contributions of all classes to compute the average metric. In a multiclass setting, micro-averaged precision, recall, and F1-score are all equal to accuracy.

## Scikit-learn Evaluation Tools:

*   `sklearn.metrics.accuracy_score(y_true, y_pred)`
*   `sklearn.metrics.precision_score(y_true, y_pred)`
*   `sklearn.metrics.recall_score(y_true, y_pred)`
*   `sklearn.metrics.f1_score(y_true, y_pred)`
*   `sklearn.metrics.confusion_matrix(y_true, y_pred)`: Returns the confusion matrix as a NumPy array.
*   `sklearn.metrics.classification_report(y_true, y_pred)`: Provides a text report showing precision, recall, F1-score, and support for each class, as well as averages. This is a very convenient way to see multiple metrics at once.

## Python Code for Practice:

The Python script `classifiers_and_evaluation.py` in this session's directory will:
1.  Use the same simple dataset from the previous session.
2.  Train additional classifiers: `LogisticRegression` and `LinearSVC`.
3.  For each model, make predictions on the test set.
4.  Calculate and print the accuracy, precision, recall, F1-score.
5.  Display the confusion matrix.
6.  Print the comprehensive `classification_report`.
7.  Compare the performance of the different classifiers.

**(Link or reference to the `classifiers_and_evaluation.py` would go here.)**

## Next Steps:

We've now learned to train and evaluate individual classifiers. In the next session, we'll explore how to streamline the process of vectorization and classification using **Scikit-learn Pipelines** and apply this to a slightly more realistic mini-project like spam detection or sentiment analysis.

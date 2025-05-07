# Session 3.5: Building a Text Classification Pipeline & Mini-Project

Welcome to Session 3.5, the final session of Module 3! We've learned how to vectorize text, train various classifiers, and evaluate their performance. Now, we'll put it all together by learning how to use **Scikit-learn Pipelines** to streamline our workflow. We'll also apply these concepts to a slightly more realistic **mini-project**: spam detection.

## Learning Objectives:

*   Understand the purpose and benefits of using Scikit-learn `Pipeline` objects.
*   Learn how to chain multiple transformers (e.g., vectorizer) and an estimator (classifier) into a single `Pipeline`.
*   Appreciate how Pipelines help prevent data leakage from the test set during preprocessing/feature extraction.
*   Get a brief conceptual introduction to **Grid Search** for hyperparameter tuning (though full implementation is for a later, more advanced stage).
*   Work on a mini-project: Building a spam detection classifier using a standard dataset.

## Why Use Scikit-learn Pipelines?

As our machine learning workflows become more complex (e.g., multiple preprocessing steps, vectorization, classification), managing each step separately can be cumbersome and error-prone. Pipelines offer several advantages:

1.  **Convenience and Organization:**
    *   They allow you to chain multiple processing steps (transformers) and a final estimator (classifier/regressor) into a single object.
    *   This makes your code cleaner, more readable, and easier to manage.

2.  **Preventing Data Leakage (Crucial!):**
    *   When performing operations like feature scaling or vectorization (`fit_transform`), it's vital that these are "fit" *only* on the training data and then "transformed" on both the training and test data.
    *   If you fit your vectorizer (e.g., `TfidfVectorizer`) on the *entire dataset* before splitting into train/test, information from the test set (e.g., word frequencies, vocabulary) "leaks" into the training process. This can lead to overly optimistic performance estimates on your test set that don't generalize to truly unseen data.
    *   **Pipelines correctly handle this:** When a pipeline is fit on the training data, the `fit_transform` method is called sequentially on the transformers for the training data. When `predict` or `score` is called with test data, only the `transform` method of the transformers is applied to the test data, using parameters learned *only* from the training data.

3.  **Joint Parameter Selection (Grid Search):**
    *   Pipelines can be used with tools like `GridSearchCV` or `RandomizedSearchCV` to search for the best combination of hyperparameters for *all* components in the pipeline simultaneously.
    *   For example, you can tune parameters of your `TfidfVectorizer` (like `ngram_range`, `max_df`) *and* parameters of your classifier (like `alpha` for Naive Bayes, `C` for SVM) in a single grid search.

## How to Create a Pipeline:

A `Pipeline` is created by providing a list of `(name, transform_or_estimator)` tuples. Each item, except the last, must be a "transformer" (i.e., have `fit` and `transform` methods). The last item must be an "estimator" (e.g., a classifier).

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define the steps in the pipeline
# Each step is a tuple: ('name_for_this_step', estimator_object)
text_clf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), # Step 1: TF-IDF Vectorization
    ('clf', MultinomialNB())                         # Step 2: Classifier
])

# Now, you can use 'text_clf_pipeline' as if it were a single estimator:
# text_clf_pipeline.fit(X_train_raw_text, y_train)
# predictions = text_clf_pipeline.predict(X_test_raw_text)
# score = text_clf_pipeline.score(X_test_raw_text, y_test)

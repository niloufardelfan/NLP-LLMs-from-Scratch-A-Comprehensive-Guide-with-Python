from sklearn.model_selection import GridSearchCV

# Example parameter grid
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Try unigrams and unigrams+bigrams for tfidf step
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3, 1.0),        # Try different alpha values for the clf (classifier) step
}

# Create GridSearchCV object
# gs_clf = GridSearchCV(text_clf_pipeline, parameters, cv=5, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train_raw_text, y_train)

# print(f"Best score: {gs_clf.best_score_}")
# print(f"Best parameters: {gs_clf.best_params_}")
# The gs_clf object now acts as the best found classifier

# Module_03_Feature_Engineering_ML_for_NLP/Session_3.3_Intro_Text_Classification/text_classification_naive_bayes.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np # For array manipulation if needed

# --- 1. Sample Data Collection ---
# For this introductory example, we'll create a very small, simple dataset.
# In real-world scenarios, you'd load data from files (CSV, JSON, text files per category).

# Documents (X - features)
documents = [
    "This is a fantastic movie, loved it!",               # Positive
    "Absolutely wonderful experience, highly recommended.", # Positive
    "The plot was engaging and the actors were brilliant.",# Positive
    "I really enjoyed this film.",                         # Positive
    "What a terrible waste of time.",                      # Negative
    "I hated every moment of it.",                         # Negative
    "The storyline was boring and predictable.",           # Negative
    "Would not recommend this to anyone.",                 # Negative
    "It was an okay movie, neither good nor bad.",         # Neutral (or could be excluded for binary)
    "The acting was decent but the script lacked depth."   # Neutral
]

# Labels (y - target)
# 0 for Negative, 1 for Positive, 2 for Neutral (for multiclass)
# For a binary example, we could just use 0 for Negative, 1 for Positive
# Let's start with a binary classification problem: Positive vs. Negative
binary_documents = documents[:4] + documents[4:8] # Select 4 positive, 4 negative
binary_labels = [1, 1, 1, 1, 0, 0, 0, 0]      # 1 for Positive, 0 for Negative

print("--- Sample Binary Classification Data ---")
for i in range(len(binary_documents)):
    print(f"Document: \"{binary_documents[i]}\" \tLabel: {binary_labels[i]}")


# --- 2. Text Preprocessing & 3. Feature Extraction (Vectorization) ---
# We'll use TF-IDF for feature extraction.
# Preprocessing like lowercasing is handled by TfidfVectorizer by default.
# We can add stop_words='english' for basic stop word removal.
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the documents and transform them into a TF-IDF matrix (X)
X_features = vectorizer.fit_transform(binary_documents)
y_labels = np.array(binary_labels) # Convert labels to a NumPy array

print(f"\nShape of feature matrix X: {X_features.shape} (documents, features/vocabulary_size)")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
# print(f"Feature names: {vectorizer.get_feature_names_out()}")
# print(f"\nTF-IDF Matrix (sparse):\n{X_features}")


# --- 4. Splitting Data into Training and Testing Sets ---
# This is crucial to evaluate how well our model generalizes to unseen data.
# test_size: proportion of the dataset to include in the test split (e.g., 0.25 for 25%)
# random_state: ensures reproducibility. The split is done randomly, so setting a seed
#               makes sure you get the same split every time you run the code.
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_labels,
    test_size=0.25, # Use 25% of data for testing (2 samples in this small dataset)
    random_state=42   # The answer to life, the universe, and everything. Ensures same split.
)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

print(f"\nTraining labels (y_train): {y_train}")
print(f"Test labels (y_test): {y_test}")


# --- 5. Model Selection & 6. Model Training ---
# We'll use Multinomial Naive Bayes (MNB), a good baseline for text classification.
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train, y_train)
print("\nMultinomial Naive Bayes model trained successfully.")


# --- 7. Making Predictions on the Test Set ---
# (Formal evaluation will be covered in the next session)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) # Get probability estimates

print("\n--- Predictions on Test Data ---")
# Let's see which documents ended up in the test set for context (not typical in real eval)
# This requires us to re-vectorize the original documents that correspond to X_test
# For simplicity here, we'll just show the predicted and true labels.

# Find original documents for X_test (this is tricky after shuffling and sparse matrix transformation)
# Instead, let's just show the predictions.
# A better way to see original test documents is to split documents BEFORE vectorization.
# documents_train, documents_test, labels_train_orig, labels_test_orig = train_test_split(
#     binary_documents, binary_labels, test_size=0.25, random_state=42
# )
# Then vectorize documents_train and documents_test separately (fit_transform on train, transform on test).
# We'll cover this proper workflow in the pipeline session.

for i in range(len(y_test)):
    print(f"Test Sample {i+1}:")
    print(f"  Predicted Label: {y_pred[i]} (0=Negative, 1=Positive)")
    print(f"  Actual Label   : {y_test[i]}")
    print(f"  Predicted Probabilities [P(Neg), P(Pos)]: [{y_pred_proba[i][0]:.2f}, {y_pred_proba[i][1]:.2f}]")

# Basic accuracy check for this small example
correct_predictions = np.sum(y_pred == y_test)
total_test_samples = len(y_test)
accuracy = correct_predictions / total_test_samples if total_test_samples > 0 else 0
print(f"\nBasic Accuracy on this small test set: {correct_predictions}/{total_test_samples} = {accuracy:.2f}")


# --- Predicting on New, Unseen Data ---
print("\n--- Predicting on New Unseen Data ---")
new_texts = [
    "This movie was absolutely fantastic and a joy to watch.",
    "A complete disaster, I want my money back.",
    "It was okay, not great but not bad either." # More neutral
]

# IMPORTANT: New data must be transformed using the SAME vectorizer that was `fit` on the training data.
new_texts_features = vectorizer.transform(new_texts)

new_predictions = model.predict(new_texts_features)
new_predictions_proba = model.predict_proba(new_texts_features)

for i in range(len(new_texts)):
    print(f"\nNew Text: \"{new_texts[i]}\"")
    print(f"  Predicted Label: {new_predictions[i]}")
    print(f"  Predicted Probabilities: [P(Neg)={new_predictions_proba[i][0]:.3f}, P(Pos)={new_predictions_proba[i][1]:.3f}]")

print("\nNote: This is a very small dataset. Performance and probability scores")
print("are illustrative. Real-world applications require much larger datasets")
print("and more rigorous evaluation (covered next!).")

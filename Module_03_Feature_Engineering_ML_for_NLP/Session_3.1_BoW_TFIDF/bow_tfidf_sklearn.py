from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd # For better display of matrices

# Sample corpus of documents
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document again?" # Added for variation
]

print("Original Corpus:")
for i, doc in enumerate(corpus):
    print(f"Document {i+1}: \"{doc}\"")

print("\n--- 1. Bag-of-Words (BoW) with CountVectorizer ---")

# Initialize CountVectorizer
# Common parameters:
#  - stop_words='english': removes common English stop words
#  - min_df: minimum document frequency (float for percentage, int for count)
#  - max_df: maximum document frequency
#  - ngram_range: (min_n, max_n) for n-grams (covered in next session)
count_vectorizer = CountVectorizer()

# Fit the vectorizer to the corpus (learns vocabulary) and transform the corpus
bow_matrix = count_vectorizer.fit_transform(corpus)

# Get the learned vocabulary
# This is a dictionary where keys are terms and values are their column indices in the matrix
vocabulary = count_vectorizer.vocabulary_
print(f"\nLearned Vocabulary (term: index):\n{vocabulary}")

# Get the feature names (ordered by their column index)
feature_names_bow = count_vectorizer.get_feature_names_out()
print(f"\nFeature Names (in matrix column order):\n{feature_names_bow}")

# The bow_matrix is a sparse matrix (efficient for mostly zero matrices)
print(f"\nBoW Matrix (sparse format):\n{bow_matrix}")
print(f"Shape of BoW Matrix: {bow_matrix.shape} (documents, vocabulary_size)")

# To view it as a dense array (for small examples):
dense_bow_matrix = bow_matrix.toarray()
print(f"\nBoW Matrix (dense format):\n{dense_bow_matrix}")

# For better readability, use pandas DataFrame
df_bow = pd.DataFrame(dense_bow_matrix, columns=feature_names_bow, index=[f"Doc{i+1}" for i in range(len(corpus))])
print(f"\nBoW Matrix (DataFrame for readability):\n{df_bow}")


print("\n\n--- 2. TF-IDF (Term Frequency-Inverse Document Frequency) with TfidfVectorizer ---")

# Initialize TfidfVectorizer
# It has similar parameters to CountVectorizer (stop_words, min_df, max_df, etc.)
# Additional parameters related to TF-IDF calculation:
#  - use_idf=True (default): Enable inverse-document-frequency reweighting.
#  - smooth_idf=True (default): Smooth idf weights by adding one to document frequencies.
#  - norm='l2' (default): Normalize term vectors to unit length (L2 norm).
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Vocabulary and feature names will be the same as CountVectorizer if parameters are similar
# and no stop words or df pruning is applied differently.
# For TF-IDF, the feature names essentially refer to the terms whose TF-IDF scores are computed.
feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
print(f"\nFeature Names for TF-IDF (should match BoW if same preprocessing):\n{feature_names_tfidf}")
# Vocabulary:
# print(f"\nVocabulary (TF-IDF Vectorizer):\n{tfidf_vectorizer.vocabulary_}")


# The tfidf_matrix is also a sparse matrix
print(f"\nTF-IDF Matrix (sparse format):\n{tfidf_matrix}")
print(f"Shape of TF-IDF Matrix: {tfidf_matrix.shape}")

# To view it as a dense array:
dense_tfidf_matrix = tfidf_matrix.toarray()
print(f"\nTF-IDF Matrix (dense format):\n{dense_tfidf_matrix}")

# For better readability, use pandas DataFrame
df_tfidf = pd.DataFrame(dense_tfidf_matrix, columns=feature_names_tfidf, index=[f"Doc{i+1}" for i in range(len(corpus))])
print(f"\nTF-IDF Matrix (DataFrame for readability):\n{df_tfidf}")

print("\n--- Observations ---")
print("1. BoW Matrix contains raw term counts.")
print("2. TF-IDF Matrix contains weighted scores. Common words occurring in many documents (like 'document', 'is', 'the')")
print("   will have their scores dampened by the IDF component compared to rarer words, even if their TF is high.")
print("   For instance, compare the values for 'the' or 'document' in BoW vs TF-IDF across different docs.")
print("   Words unique to a document or fewer documents will get higher TF-IDF scores if their TF is reasonable.")

# Example: IDF scores for each term
# print("\nIDF scores for each term:")
# for term, score in zip(feature_names_tfidf, tfidf_vectorizer.idf_):
#     print(f"  Term: '{term}', IDF: {score:.4f}")

print("\n--- Example with Stop Words using TfidfVectorizer ---")
tfidf_vectorizer_stopwords = TfidfVectorizer(stop_words='english')
tfidf_matrix_stopwords = tfidf_vectorizer_stopwords.fit_transform(corpus)
feature_names_stopwords = tfidf_vectorizer_stopwords.get_feature_names_out()
df_tfidf_stopwords = pd.DataFrame(tfidf_matrix_stopwords.toarray(), columns=feature_names_stopwords, index=[f"Doc{i+1}" for i in range(len(corpus))])
print(f"\nTF-IDF Matrix (with English stop words removed):\n{df_tfidf_stopwords}")
print("Notice how common words like 'this', 'is', 'the', 'and' are now removed from the features.")

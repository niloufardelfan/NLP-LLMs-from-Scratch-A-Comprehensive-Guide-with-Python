# Session 3.1: Text Representation - Bag-of-Words (BoW) and TF-IDF

Welcome to Module 3, Session 1! Now that we know how to preprocess text, the next crucial step is to convert this cleaned text into a numerical format that machine learning algorithms can understand. This session focuses on two fundamental techniques for text representation: **Bag-of-Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**.

## Learning Objectives:

*   Understand why raw text needs to be converted into numerical features for machine learning.
*   Grasp the concept of the Bag-of-Words (BoW) model, including vocabulary creation and term frequency counting.
*   Understand the components of Term Frequency-Inverse Document Frequency (TF-IDF):
    *   Term Frequency (TF)
    *   Inverse Document Frequency (IDF)
    *   The TF-IDF weighting scheme.
*   Learn how to use Scikit-learn's `CountVectorizer` (for BoW) and `TfidfVectorizer` (for TF-IDF) to transform text data into numerical matrices.
*   Appreciate the strengths and limitations of these representation methods.

## Why Numerical Representation?

Machine learning algorithms, especially traditional ones like Naive Bayes, Logistic Regression, and SVMs, operate on numerical data. They cannot directly process raw strings of text. Therefore, we need methods to convert collections of text documents into numerical feature vectors. This process is often called **feature extraction** or **text vectorization**.

## 1. Bag-of-Words (BoW) Model:

*   **Concept:** The Bag-of-Words model is a simple and intuitive way to represent text data. It describes the occurrence of each word within a document. It involves two main steps:
    1.  **Vocabulary Creation:** Building a vocabulary of all unique words present in the entire collection of documents (corpus).
    2.  **Document Vectorization:** For each document, creating a numerical vector where each component corresponds to a word in the vocabulary. The value of the component is typically the **frequency of that word in the document** (term frequency).

*   **Analogy:** Imagine you have a document and a dictionary (your vocabulary). You take all the words from the document, put them into a "bag," and count how many times each dictionary word appears in your bag. The order of words is disregarded, hence the "bag" analogy.

*   **Example:**
    Consider the following two short documents:
    *   Document 1: "The cat sat on the mat."
    *   Document 2: "The dog played with the ball."

    1.  **Vocabulary (after preprocessing like lowercasing, punctuation removal):**
        `{"the", "cat", "sat", "on", "mat", "dog", "played", "with", "ball"}`
        Let's order it: `["ball", "cat", "dog", "mat", "on", "played", "sat", "the", "with"]` (9 unique words)

    2.  **Document Vectors (using term frequencies):**
        *   **Document 1 Vector:** How many times does each vocabulary word appear in "the cat sat on the mat"?
            `[0, 1, 0, 1, 1, 0, 1, 2, 0]`
            (0 'ball', 1 'cat', 0 'dog', 1 'mat', 1 'on', 0 'played', 1 'sat', 2 'the', 0 'with')
        *   **Document 2 Vector:** How many times does each vocabulary word appear in "the dog played with the ball"?
            `[1, 0, 1, 0, 0, 1, 0, 2, 1]`

*   **Implementation with Scikit-learn (`CountVectorizer`):**
    Scikit-learn's `CountVectorizer` handles vocabulary building and term counting.
    *   `fit(corpus)`: Learns the vocabulary from the corpus.
    *   `transform(corpus)`: Converts the corpus into a document-term matrix (a sparse matrix where rows are documents and columns are vocabulary terms).
    *   `fit_transform(corpus)`: Combines both steps.

*   **Pros of BoW:**
    *   Simple to understand and implement.
    *   Provides a basic numerical representation that works surprisingly well for some tasks like document classification.
*   **Cons of BoW:**
    *   **Loss of Word Order:** Ignores grammar and the sequence of words, which can be crucial for meaning (e.g., "man bites dog" vs. "dog bites man" would have similar BoW vectors if vocabulary is small).
    *   **High Dimensionality:** The vocabulary can become very large for big corpora, leading to very high-dimensional vectors (curse of dimensionality).
    *   **Sparsity:** Most documents will only contain a small subset of the total vocabulary, so the vectors will be mostly zeros (sparse).
    *   **Doesn't Account for Word Importance:** Common words like "the" will have high counts but might not be very informative for distinguishing between documents. This is where TF-IDF comes in.

## 2. Term Frequency-Inverse Document Frequency (TF-IDF):

*   **Concept:** TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It addresses the issue of BoW where common words can dominate less frequent but more informative words.
*   **It consists of two parts:**

    *   **a. Term Frequency (TF):**
        *   **Definition:** Measures how frequently a term `t` appears in a document `d`.
        *   **Formula:** There are several ways to calculate TF:
            1.  **Raw Count:** `TF(t, d) = (Number of times term t appears in document d)` (This is what BoW uses).
            2.  **Normalized Frequency:** `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)` (To prevent a bias towards longer documents).
            3.  **Logarithmic Scaling:** `TF(t, d) = log(1 + raw count of t in d)` (To dampen the effect of very high counts).
        *   **Intuition:** Words that appear more often in a document are likely more important for that document.

    *   **b. Inverse Document Frequency (IDF):**
        *   **Definition:** Measures how much information a word provides, i.e., whether it's common or rare across all documents in the corpus.
        *   **Formula:**
            `IDF(t, D) = log( (Total number of documents in corpus D) / (Number of documents containing term t) )`
            Often, `1` is added to the denominator (and sometimes to the numerator's total document count) to prevent division by zero if a term is not in any document or appears in all documents (IDF smoothing).
            `IDF(t, D) = log( (N + 1) / (df(t) + 1) ) + 1` (Scikit-learn's default smoothing)
            where `N` is total documents, `df(t)` is document frequency of term `t`.
        *   **Intuition:**
            *   If a word appears in many documents (e.g., "the", "a"), its IDF will be low, indicating it's less informative.
            *   If a word appears in few documents, its IDF will be high, indicating it's more unique and potentially more discriminative.

    *   **c. TF-IDF Score:**
        *   **Formula:** `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`
        *   **Value:** The TF-IDF score is higher for terms that have a high frequency within a specific document (high TF) AND are rare across the entire corpus (high IDF). It's lower for terms that are common across the corpus or rare/absent in the specific document.

*   **Implementation with Scikit-learn (`TfidfVectorizer`):**
    `TfidfVectorizer` combines the functionality of `CountVectorizer` with TF-IDF transformation.
    *   It first computes term counts (like `CountVectorizer`).
    *   Then, it calculates TF and IDF values and computes the final TF-IDF scores.
    *   It also has parameters for `min_df` (ignore terms with document frequency lower than threshold), `max_df` (ignore terms with document frequency higher than threshold - good for corpus-specific stop words), `ngram_range`, etc.

*   **Pros of TF-IDF:**
    *   Simple to compute.
    *   Gives more weight to terms that are important in a specific document but not common across all documents.
    *   Often improves performance over simple BoW for tasks like text classification and information retrieval.
*   **Cons of TF-IDF:**
    *   Still suffers from the loss of word order and semantic meaning (like BoW).
    *   Can still lead to high-dimensional sparse vectors.
    *   IDF is corpus-dependent; adding new documents can change IDF scores.

## Python Code for Practice:

The Python script `bow_tfidf_sklearn.py` in this session's directory demonstrates:
1.  How to use `CountVectorizer` to create Bag-of-Words representations.
2.  How to inspect the vocabulary and the document-term matrix.
3.  How to use `TfidfVectorizer` to create TF-IDF representations.
4.  How to observe the difference in feature values between BoW and TF-IDF.

**(Link or reference to the `bow_tfidf_sklearn.py` would go here.)**

## Next Steps:

While BoW and TF-IDF are foundational, their limitation of ignoring word order is significant. In the next session, we'll introduce **N-grams**, which help capture some local word order, and touch upon basic concepts of **Language Modeling**.

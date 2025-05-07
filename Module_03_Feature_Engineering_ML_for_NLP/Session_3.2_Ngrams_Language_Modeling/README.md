# Session 3.2: N-grams and Basic Language Modeling

Welcome to Session 3.2! In the previous session, we learned about Bag-of-Words (BoW) and TF-IDF, which represent text based on word frequencies. However, a major limitation of these models is that they disregard word order. This session introduces **N-grams**, a simple yet effective way to incorporate some local word order, and provides a conceptual introduction to **Language Modeling**.

## Learning Objectives:

*   Understand the limitations of BoW/TF-IDF related to word order.
*   Define N-grams (unigrams, bigrams, trigrams, etc.) and understand how they capture local context.
*   Generate N-grams using NLTK.
*   Utilize the `ngram_range` parameter in Scikit-learn's vectorizers (`CountVectorizer`, `TfidfVectorizer`) to create features from N-grams.
*   Get a conceptual introduction to Language Modeling (LM) – the task of predicting the next word.
*   Briefly understand how N-grams form the basis of simple statistical language models and recognize the sparsity problem associated with them.

## Limitations of Word Order Disregard (in BoW/TF-IDF):

Models like BoW and TF-IDF treat text as an unordered collection of words. This means sentences like:
*   "The quick brown fox jumps over the lazy dog."
*   "The lazy dog jumps over the quick brown fox."
would have very similar (or identical, if simple BoW counts are used without stop words) vector representations. However, the meaning can be subtly or significantly different depending on word order, especially for tasks involving understanding nuances, negations, or specific phrases.

Consider:
*   "not good" vs. "good"
*   "New York" (a specific entity) vs. "new" and "york" as separate words.

## 1. N-grams: Capturing Local Word Order

*   **Definition:** An N-gram is a contiguous sequence of `n` items from a given sample of text or speech. The items can be characters, syllables, or (most commonly in NLP) words.
    *   **Unigram (1-gram):** A single word (e.g., "the", "quick", "fox"). This is what BoW and TF-IDF use by default.
    *   **Bigram (2-gram):** A sequence of two adjacent words (e.g., "the quick", "quick brown", "brown fox").
    *   **Trigram (3-gram):** A sequence of three adjacent words (e.g., "the quick brown", "quick brown fox").
    *   ...and so on (4-grams, 5-grams, etc.).

*   **How N-grams Help:** By considering sequences of words, N-grams capture local word order and can represent phrases.
    *   For "New York", the bigram "New York" can be a single feature.
    *   For "not good", the bigram "not good" can capture the negation.

*   **Using N-grams as Features:** Instead of just using individual words as features in BoW or TF-IDF, we can use N-grams. For example, a document could be represented by counts or TF-IDF scores of unigrams, bigrams, and trigrams found within it. This increases the vocabulary size but can add valuable contextual information.

*   **Generating N-grams with NLTK:**
    NLTK provides a utility function `nltk.util.ngrams()` for this.
    ```python
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams

    text = "This is a sample sentence."
    tokens = word_tokenize(text.lower())
    bigrams = list(ngrams(tokens, 2)) # list() converts the generator
    # Output: [('this', 'is'), ('is', 'a'), ('a', 'sample'), ('sample', 'sentence'), ('sentence', '.')]
    trigrams = list(ngrams(tokens, 3))
    ```

*   **Using N-grams in Scikit-learn Vectorizers:**
    `CountVectorizer` and `TfidfVectorizer` have an `ngram_range` parameter.
    *   `ngram_range=(min_n, max_n)`: This tells the vectorizer to use N-grams where `n` is between `min_n` and `max_n` (inclusive).
    *   `ngram_range=(1, 1)`: Only unigrams (default).
    *   `ngram_range=(2, 2)`: Only bigrams.
    *   `ngram_range=(1, 2)`: Use both unigrams and bigrams.
    *   `ngram_range=(1, 3)`: Use unigrams, bigrams, and trigrams.

    When N-grams are used, the "vocabulary" learned by the vectorizer will consist of these N-gram sequences.

## 2. Introduction to Language Modeling (LM):

*   **Concept:** Language Modeling is the task of assigning a probability to a sequence of words (a sentence) or, more commonly, predicting the probability of the next word in a sequence given the preceding words.
    `P(w_n | w_1, w_2, ..., w_{n-1})` - Probability of word `w_n` given previous words.

*   **Applications of LM:**
    *   **Speech Recognition:** Helps choose between phonetically similar words (e.g., "recognize speech" vs. "wreck a nice beach").
    *   **Machine Translation:** Helps generate fluent and grammatically correct translations.
    *   **Spell Correction & Grammar Checking:** Suggests more probable sequences.
    *   **Text Generation (Autocompletion, Predictive Text):** Suggesting the next word or completing a sentence.
    *   **Information Retrieval:** Ranking documents based on the probability of generating the query.

## 3. N-gram Language Models (Statistical LMs):

*   **Basic Idea:** A simple way to build a language model is to use N-gram frequencies from a large training corpus.
*   **Markov Assumption:** To make prediction feasible, N-gram models often make a Markov assumption: the probability of the next word depends only on a fixed number (`k`) of preceding words.
    *   For a bigram model (`k=1`): `P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-1})`
    *   For a trigram model (`k=2`): `P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-2}, w_{n-1})`

*   **Calculating Probabilities (Maximum Likelihood Estimation - MLE):**
    *   **Bigram Probability:**
        `P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})`
        (How many times does `w_{i-1}` followed by `w_i` appear, divided by how many times `w_{i-1}` appears?)
    *   **Trigram Probability:**
        `P(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})`

*   **Sparsity Problem:**
    *   A major issue with N-gram models is data sparsity. Many valid N-grams (especially for larger `n`) will not appear in the training corpus. This means their count will be zero, leading to a zero probability for any sentence containing them.
    *   For example, if the trigram "learn natural language" never appeared, its probability would be zero, even if "learn natural" and "natural language" appeared separately.
    *   **Smoothing Techniques:** Various techniques (e.g., Add-1 (Laplace) smoothing, Add-k smoothing, Kneser-Ney smoothing) are used to address this by redistributing some probability mass from seen N-grams to unseen N-grams.

*   **Evaluation of LMs:**
    *   **Perplexity:** A common intrinsic metric. Lower perplexity indicates a better model (it's less "surprised" by the test data).

While traditional N-gram LMs have been largely superseded by neural network-based LMs (like RNNs and Transformers, which we'll cover later), understanding N-grams is fundamental as they still play a role in feature engineering and provide intuition for sequence modeling.

## Python Code for Practice:

The Python script `ngrams_practice.py` in this session's directory demonstrates:
1.  Generating N-grams (bigrams, trigrams) from a sentence using NLTK.
2.  Using the `ngram_range` parameter in Scikit-learn's `CountVectorizer` to include N-grams as features.
3.  A conceptual example of calculating bigram probabilities (this will be simplified due to the scope).

**(Link or reference to the `ngrams_practice.py` would go here.)**

## Next Steps:

Now that we can represent text numerically (using BoW, TF-IDF, and N-grams), we are ready to introduce basic **Text Classification** using traditional machine learning algorithms in the next session. We'll see how Scikit-learn makes this process accessible.

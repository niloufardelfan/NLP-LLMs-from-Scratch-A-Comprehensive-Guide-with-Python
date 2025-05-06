# Session 2.2: Text Preprocessing - Stemming & Lemmatization

Welcome to Session 2.2! We continue our exploration of text preprocessing by focusing on two important techniques for word normalization: **Stemming** and **Lemmatization**. These methods aim to reduce words to their base or root form, further helping to consolidate vocabulary and improve model performance.

## Learning Objectives:

*   Understand the concept and purpose of stemming.
*   Apply common stemming algorithms like Porter Stemmer and Snowball Stemmer using NLTK.
*   Understand the concept and purpose of lemmatization.
*   Perform lemmatization using NLTK (with WordNetLemmatizer) and spaCy.
*   Clearly differentiate between stemming and lemmatization, including their pros and cons.

## What are Stemming and Lemmatization?

Both stemming and lemmatization are techniques used to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.
For example, words like "running", "runs", "ran" might all be reduced to a common form like "run".

**Why are they important?**
*   **Vocabulary Reduction:** They map different forms of the same word to a single token. This reduces the overall vocabulary size, which can be beneficial for many NLP models by reducing sparsity and computational complexity.
*   **Improved Generalization:** By treating variants of a word as the same, models can generalize better from the training data. For example, if a model learns something about "running", it can apply that knowledge to "runs" as well if they are both reduced to "run".

## 1. Stemming:

*   **Concept:** Stemming is a somewhat crude, rule-based process of chopping off the ends (suffixes) of words to obtain a common "stem". The resulting stem may not always be a valid dictionary word.
*   **How it works:** It typically uses heuristic algorithms that identify and remove common morphological affixes. It doesn't usually consider the context of the word.
*   **Common Stemming Algorithms:**
    *   **Porter Stemmer:** One of the earliest and most well-known stemming algorithms for English. It consists of a series of cascaded rules for suffix stripping.
    *   **Snowball Stemmer (Porter2 Stemmer):** An improvement over the original Porter Stemmer, also developed by Martin Porter. It's generally considered more aggressive or effective and supports multiple languages.
    *   **Lancaster Stemmer:** A more aggressive stemmer known for producing shorter stems. It can sometimes be too aggressive and lead to over-stemming.
*   **Pros:**
    *   Computationally fast and simple to implement.
    *   Effective at reducing words to a common form, which can be good for information retrieval tasks.
*   **Cons:**
    *   **Over-stemming:** Removing too much of the word, leading to different words being reduced to the same stem (e.g., "universal", "university", "universe" might all become "univers"). This results in a loss of meaning.
    *   **Under-stemming:** Not reducing words enough, leading to related words having different stems (e.g., "data" and "datum" might not be stemmed to the same form by some stemmers).
    *   The resulting stems are often not actual words (e.g., "studies" might become "studi").
*   **Implementation (NLTK):**
    NLTK provides implementations for various stemmers.

## 2. Lemmatization:

*   **Concept:** Lemmatization is a more sophisticated process of reducing a word to its **lemma** (canonical dictionary form or citation form). The lemma is always a valid word.
*   **How it works:** It typically involves:
    *   Using a vocabulary (like a dictionary) and morphological analysis of words.
    *   Considering the **Part-of-Speech (POS)** tag of a word. For example, the lemma of "better" is "good" if it's an adjective, but "better" if it's a verb. "meeting" can be a noun (lemma: "meeting") or a verb (lemma: "meet").
*   **Pros:**
    *   Produces actual dictionary words, which are more interpretable.
    *   More accurate reduction to the base form compared to stemming, preserving more meaning.
*   **Cons:**
    *   Computationally more expensive than stemming because it often involves dictionary lookups and POS tagging.
    *   Requires more linguistic resources (dictionaries, morphological analyzers, POS taggers).
*   **Implementation:**
    *   **NLTK:** Uses `WordNetLemmatizer`, which leverages the WordNet lexical database. For best results, it's important to provide the POS tag of the word.
    *   **spaCy:** Lemmatization is an integral part of spaCy's processing pipeline. When you process text with a spaCy `nlp` object, each `Token` object has a `lemma_` attribute that provides the lemma. spaCy's lemmatizer is typically rule-based or uses look-up tables and takes POS into account.

## Stemming vs. Lemmatization - Key Differences:

| Feature         | Stemming                                       | Lemmatization                                      |
|-----------------|------------------------------------------------|----------------------------------------------------|
| **Process**     | Chops off suffixes (rule-based heuristics)     | Uses vocabulary and morphological analysis          |
| **Output**      | May not be a valid word (stem)                | Always a valid dictionary word (lemma)            |
| **Accuracy**    | Less accurate, prone to over/under-stemming  | More accurate, context-aware (with POS)          |
| **Speed**       | Faster                                         | Slower (due to lookups and POS tagging)            |
| **Complexity**  | Simpler                                        | More complex, requires more linguistic resources   |
| **When to use** | Information Retrieval, when speed is critical and slight inaccuracy is acceptable. | Text understanding tasks, chatbots, machine translation, where meaning preservation is important. |

**General Guideline:** If you need speed and the exact dictionary form is not critical, stemming might be sufficient. If accuracy and interpretability are important, and you can afford the computational overhead, lemmatization is generally preferred.

## Python Code for Practice:

The Python script `stemming_lemmatization_practice.py` in this session's directory demonstrates:
1.  How to use Porter Stemmer and Snowball Stemmer from NLTK.
2.  How to use WordNetLemmatizer from NLTK (with and without POS tags).
3.  How to access lemmas using spaCy.

**(Link or reference to the `stemming_lemmatization_practice.py` would go here.)**

Remember to download `wordnet` and `averaged_perceptron_tagger` for NLTK if you haven't already:
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4') # Open Multilingual Wordnet, often a dependency for WordNet

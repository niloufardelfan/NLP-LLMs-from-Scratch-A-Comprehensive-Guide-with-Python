# Module_03_Feature_Engineering_ML_for_NLP/Session_3.2_Ngrams_Language_Modeling/ngrams_practice.py

# NLTK for N-gram generation
import nltk
try:
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' resource not found for tokenization. Downloading...")
    nltk.download('punkt', quiet=True)
except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    exit()

# Scikit-learn for N-grams in vectorizers
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd # For better display

# Sample text
text1 = "This is a simple sentence for N-gram demonstration."
text2 = "N-grams can capture local word order information."
corpus = [text1, text2]

print("Original Corpus:")
for i, doc in enumerate(corpus):
    print(f"Document {i+1}: \"{doc}\"")


print("\n--- 1. Generating N-grams with NLTK ---")
tokens1 = word_tokenize(text1.lower()) # Tokenize and lowercase
print(f"\nTokens for Document 1: {tokens1}")

# Generate Bigrams (2-grams)
nltk_bigrams = list(ngrams(tokens1, 2)) # list() converts the ngrams generator
print(f"NLTK Bigrams (Document 1): {nltk_bigrams}")

# Generate Trigrams (3-grams)
nltk_trigrams = list(ngrams(tokens1, 3))
print(f"NLTK Trigrams (Document 1): {nltk_trigrams}")

# Generate Unigrams, Bigrams, and Trigrams
print("\nGenerating multiple n-gram types for Document 1:")
for n_val in [1, 2, 3]:
    n_gram_list = list(ngrams(tokens1, n_val))
    print(f"  NLTK {n_val}-grams: {n_gram_list[:5]} ... (showing first 5 if many)")


print("\n\n--- 2. Using N-grams with Scikit-learn's CountVectorizer ---")
print("Original Corpus for Scikit-learn:")
for i, doc in enumerate(corpus):
    print(f"Document {i+1}: \"{doc}\"")

# a. Unigrams only (default behavior)
count_vec_unigrams = CountVectorizer(ngram_range=(1, 1))
bow_unigrams = count_vec_unigrams.fit_transform(corpus)
df_unigrams = pd.DataFrame(bow_unigrams.toarray(), columns=count_vec_unigrams.get_feature_names_out(), index=["Doc1", "Doc2"])
print(f"\nBoW with Unigrams (ngram_range=(1,1)):\n{df_unigrams}")

# b. Bigrams only
count_vec_bigrams = CountVectorizer(ngram_range=(2, 2))
bow_bigrams = count_vec_bigrams.fit_transform(corpus)
df_bigrams = pd.DataFrame(bow_bigrams.toarray(), columns=count_vec_bigrams.get_feature_names_out(), index=["Doc1", "Doc2"])
print(f"\nBoW with Bigrams only (ngram_range=(2,2)):\n{df_bigrams}")
print("Notice how 'n gram' becomes a feature for Doc1.")

# c. Unigrams AND Bigrams
count_vec_uni_bi = CountVectorizer(ngram_range=(1, 2))
bow_uni_bi = count_vec_uni_bi.fit_transform(corpus)
df_uni_bi = pd.DataFrame(bow_uni_bi.toarray(), columns=count_vec_uni_bi.get_feature_names_out(), index=["Doc1", "Doc2"])
print(f"\nBoW with Unigrams AND Bigrams (ngram_range=(1,2)):\n{df_uni_bi}")
print("Features now include individual words and pairs of words.")


print("\n\n--- 3. Conceptual Introduction to N-gram Language Modeling ---")
# This is a highly simplified demonstration for conceptual understanding only.
# Real N-gram LMs involve large corpora, smoothing, and more robust probability calculations.

print("Example for Basic Bigram Probabilities (Conceptual):")
training_corpus_lm = [
    "the cat sat",
    "the cat ran",
    "the dog ran",
    "a dog sat"
]
print(f"Training Corpus for LM: {training_corpus_lm}")

# Preprocess and get all bigrams and unigram counts from the training corpus
all_tokens_lm = []
for sentence in training_corpus_lm:
    all_tokens_lm.extend(word_tokenize(sentence.lower())) # Add start/end symbols for proper LM

# For simplicity, let's use a helper like NLTK's ConditionalFreqDist for counts
# A more manual way for demonstration:
unigram_counts = {}
bigram_counts = {}

for sentence in training_corpus_lm:
    tokens = ["<s>"] + word_tokenize(sentence.lower()) + ["</s>"] # Add start/end symbols
    for i in range(len(tokens)):
        unigram_counts[tokens[i]] = unigram_counts.get(tokens[i], 0) + 1
        if i > 0:
            bigram = (tokens[i-1], tokens[i])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

print(f"\nUnigram Counts (simplified): {unigram_counts}")
print(f"Bigram Counts (simplified): {bigram_counts}")

# Calculate P(word | "cat") -> e.g., P("sat" | "cat")
word_given_cat_options = {}
target_prev_word = "cat"

if target_prev_word in unigram_counts:
    count_prev_word = unigram_counts[target_prev_word]
    print(f"\nCalculating P(word | '{target_prev_word}'): Count('{target_prev_word}') = {count_prev_word}")

    for bigram, count_bg in bigram_counts.items():
        if bigram[0] == target_prev_word:
            next_word = bigram[1]
            probability = count_bg / count_prev_word
            word_given_cat_options[next_word] = probability
            print(f"  P('{next_word}' | '{target_prev_word}') = {count_bg}/{count_prev_word} = {probability:.2f}")

    if not word_given_cat_options:
        print(f"  No bigrams found starting with '{target_prev_word}' to calculate probabilities.")
else:
    print(f"'{target_prev_word}' not found in unigram counts.")


print("\nKey LM Concepts Illustrated (Simplified):")
print("1. N-gram models estimate the probability of the next word based on previous N-1 words.")
print("2. Probabilities are typically derived from counts in a training corpus.")
print("3. Sparsity is a major issue: if an N-gram (e.g., 'cat meowed') never appeared in training,")
print("   its probability would be 0 without smoothing techniques (not shown here).")
print("4. Real LMs use large corpora and smoothing (e.g., Laplace, Kneser-Ney).")

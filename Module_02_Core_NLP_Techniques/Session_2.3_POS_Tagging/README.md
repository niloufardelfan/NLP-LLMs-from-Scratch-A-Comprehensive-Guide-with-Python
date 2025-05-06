# Session 2.3: Part-of-Speech (POS) Tagging

Welcome to Session 2.3! In this session, we'll explore **Part-of-Speech (POS) Tagging**, a fundamental NLP task that involves assigning a grammatical category (like noun, verb, adjective, adverb, etc.) to each word in a text.

## Learning Objectives:

*   Understand what Part-of-Speech tagging is and its significance in NLP.
*   Learn about common POS tagsets (e.g., Penn Treebank, Universal Dependencies).
*   Perform POS tagging using NLTK.
*   Perform POS tagging using spaCy and explore its token attributes.
*   Appreciate how POS information can be used in downstream applications (e.g., improving lemmatization, information extraction).

## What is Part-of-Speech (POS) Tagging?

POS tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context. In simpler terms, it's about identifying the grammatical role each word plays in a sentence.

**Examples:**
*   "The" is a Determiner (DT)
*   "cat" is a Noun (NN)
*   "sat" is a Verb, past tense (VBD)
*   "on" is a Preposition (IN)
*   "mat" is a Noun (NN)

Sentence: `The cat sat on the mat.`
POS Tags: `DT NN VBD IN DT NN .` (Example using Penn Treebank tags)

## Why is POS Tagging Important?

POS information is a crucial piece of syntactic information that helps in understanding the structure and meaning of a sentence. It's a foundational step for many higher-level NLP tasks:

1.  **Lemmatization:** As seen in the previous session, knowing the POS tag (e.g., whether "meeting" is a noun or a verb) leads to more accurate lemmatization.
2.  **Named Entity Recognition (NER):** POS tags can be strong features for NER systems (e.g., proper nouns often indicate names).
3.  **Information Extraction:** Identifying nouns, verbs, and adjectives can help in extracting relationships and facts from text (e.g., Subject-Verb-Object patterns).
4.  **Parsing and Syntactic Analysis:** POS tags are the input for syntactic parsers that aim to determine the grammatical structure of sentences.
5.  **Question Answering:** Understanding the grammatical role of words helps in interpreting questions and finding answers.
6.  **Machine Translation:** Preserving grammatical structure often relies on correct POS tagging in both source and target languages.
7.  **Word Sense Disambiguation:** The POS tag can help disambiguate words with multiple meanings (e.g., "book" as a noun vs. "book" as a verb).

## POS Tagsets:

A POS tagset is a collection of tags used to mark the parts of speech. Different tagsets exist, varying in granularity and conventions.

1.  **Penn Treebank Tagset:**
    *   Widely used, especially in older NLP research and with NLTK.
    *   Quite detailed, with around 36-45 tags (e.g., `NN` for singular noun, `NNS` for plural noun, `NNP` for proper singular noun, `VB` for verb base form, `VBD` for verb past tense, `JJ` for adjective).
    *   You can often find a list of these tags by typing `nltk.help.upenn_tagset()` in a Python interpreter after importing NLTK.

2.  **Universal Dependencies (UD) Tagset:**
    *   A more modern effort to create a consistent POS tagging scheme (and dependency grammar annotation) across many languages.
    *   Simpler, with around 17 coarse-grained POS tags (e.g., `NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN`, `ADP` (adposition/preposition), `DET`).
    *   spaCy's default `token.pos_` attribute often aligns with or is easily mappable to UD tags. spaCy's `token.tag_` attribute provides a more fine-grained tag, which is usually the tagset the underlying model was trained on (often similar to Penn Treebank for English models).

## How POS Tagging Works (Brief Overview):

POS taggers have evolved:
*   **Rule-Based Taggers:** Early systems used hand-crafted rules based on lexical information and word endings.
*   **Stochastic/Probabilistic Taggers:** These models learn from annotated training data (corpora where words are already tagged with their POS).
    *   **Hidden Markov Models (HMMs):** A common approach where the POS tags are hidden states and words are observations. The model calculates the most likely sequence of tags for a given sequence of words.
    *   **Maximum Entropy Models (MaxEnt):** Another statistical method.
*   **Deep Learning Taggers:** Modern state-of-the-art taggers often use neural networks (e.g., LSTMs, BiLSTMs, Transformers) trained on large annotated datasets. They can capture complex contextual information effectively.

## POS Tagging with NLTK:

*   NLTK's primary POS tagging function is `nltk.pos_tag()`.
*   It takes a list of tokens (words) as input.
*   By default, it uses a pre-trained tagger called the "averaged_perceptron_tagger", which is based on the Perceptron algorithm and trained on sections of the Wall Street Journal corpus (using the Penn Treebank tagset).

```python
import nltk
from nltk.tokenize import word_tokenize

# Ensure 'averaged_perceptron_tagger' is downloaded
# nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(text)
tagged_tokens_nltk = nltk.pos_tag(tokens)
# Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ...]

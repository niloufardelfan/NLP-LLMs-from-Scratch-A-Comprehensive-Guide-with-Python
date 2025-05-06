# Session 1.4: Introduction to Core NLP Libraries: NLTK & spaCy

Welcome to Session 1.4! Having built a solid Python foundation, we now introduce two cornerstone libraries in the NLP ecosystem: **NLTK (Natural Language Toolkit)** and **spaCy**. We'll cover their installation, core philosophies, and how to perform basic NLP tasks like tokenization with each.

## Learning Objectives:

*   Install and set up the NLTK and spaCy libraries.
*   Understand the distinct design philosophies and typical use cases for NLTK and spaCy.
*   Perform fundamental NLP tasks: sentence splitting and word tokenization using both NLTK and spaCy.
*   Appreciate the data structures and basic processing pipelines of each library.

## Key Concepts Covered:

1.  **NLTK (Natural Language Toolkit):**
    *   **What it is:** NLTK is a comprehensive and pioneering open-source Python library for NLP. It's widely used in academic research and education due to its extensive range of tools, algorithms, and lexical resources (corpora).
    *   **Philosophy & Strengths:**
        *   **Educational & Research-Oriented:** Excellent for learning NLP concepts from the ground up, experimenting with various algorithms, and accessing diverse linguistic data.
        *   **Modular Design:** Provides many individual components (tokenizers, stemmers, taggers, parsers, etc.) that you can combine.
        *   **Extensive Resources:** Comes with over 50 corpora and lexical resources (like WordNet, a large lexical database of English).
    *   **Installation:**
        ```bash
        pip install nltk
        ```
    *   **Downloading NLTK Data:** NLTK itself is the framework; many of its functionalities require separate data packages (corpora, pre-trained models for tokenization, tagging, etc.).
        *   You can use the NLTK Downloader:
            ```python
            import nltk
            # nltk.download() # This opens a graphical interface to select packages.
            ```
        *   Or download specific packages programmatically (recommended for scripts):
            ```python
            import nltk
            try:
                # Check if 'punkt' (for tokenization) is available, download if not.
                nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError:
                print("Downloading NLTK 'punkt' resource...")
                nltk.download('punkt', quiet=True) # quiet=True suppresses verbose output

            # Similarly for other common resources:
            # nltk.download('stopwords')
            # nltk.download('averaged_perceptron_tagger')
            # nltk.download('wordnet')
            # nltk.download('omw-1.4') # Often needed by wordnet
            ```
    *   **Core NLTK Operations:**
        *   **Sentence Tokenization (Splitting text into sentences):**
            NLTK's `sent_tokenize` function, often based on the Punkt sentence tokenizer model, is effective for this.
            ```python
            from nltk.tokenize import sent_tokenize
            text = "Hello Mr. Smith. How are you today? N.L.P. is great!"
            sentences = sent_tokenize(text)
            # sentences will be: ['Hello Mr. Smith.', 'How are you today?', 'N.L.P. is great!']
            ```
        *   **Word Tokenization (Splitting sentences/text into words/tokens):**
            NLTK's `word_tokenize` is a common choice, often adhering to conventions like the Penn Treebank standard. It handles punctuation and contractions reasonably well.
            ```python
            from nltk.tokenize import word_tokenize
            sentence = "Don't hesitate to ask questions."
            words = word_tokenize(sentence)
            # words will be: ['Do', "n't", 'hesitate', 'to', 'ask', 'questions', '.']
            ```

2.  **spaCy:**
    *   **What it is:** spaCy is a modern, highly efficient, and production-ready open-source library for NLP in Python. It's designed for speed and ease of use, particularly for building real-world applications.
    *   **Philosophy & Strengths:**
        *   **Production-Oriented & Opinionated:** Focuses on providing robust, well-optimized solutions for common NLP tasks. It often provides "one best way" to do things, making it easier to get started with production-quality pipelines.
        *   **Speed and Efficiency:** Implemented in Cython, making it very fast and memory-efficient, suitable for processing large volumes of text.
        *   **Pre-trained Statistical Models:** Comes with excellent pre-trained models for various languages, which include support for tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and word vectors out-of-the-box.
        *   **Rich Data Structures:** Works with `Doc`, `Token`, and `Span` objects that hold a wealth of linguistic annotations.
    *   **Installation:**
        ```bash
        pip install spacy
        ```
    *   **Downloading Language Models:** spaCy's power comes from its statistical models. You need to download these separately after installing spaCy.
        *   For example, to download the small English model:
            ```bash
            python -m spacy download en_core_web_sm
            ```
        *   Other models include `en_core_web_md` (medium, with word vectors), `en_core_web_lg` (large, with more accurate word vectors), and models for many other languages (e.g., `de_core_news_sm` for German).
    *   **Loading a Language Model & Processing Text:**
        The `nlp` object, created by loading a model, is your main entry point for processing text.
        ```python
        import spacy

        # Load the pre-trained English model
        # Ensure you've downloaded it first: python -m spacy download en_core_web_sm
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found.")
            print("Please download it: python -m spacy download en_core_web_sm")
            exit() # Or handle appropriately

        text = "Apple Inc. is looking at buying U.K. startup for $1 billion."
        doc = nlp(text) # This processes the text and creates a Doc object
        ```
        The `doc` object is a container for a sequence of `Token` objects and now holds all the annotations produced by the pipeline (e.g., POS tags, entities).
    *   **Core spaCy Operations:**
        *   **Sentence Tokenization:** spaCy's processing pipeline (typically including a dependency parser) automatically segments the text into sentences. These are accessible via the `doc.sents` attribute.
            ```python
            # Continuing from above
            for i, sentence_span in enumerate(doc.sents):
                print(f"Sentence {i+1}: {sentence_span.text}")
            # Output example:
            # Sentence 1: Apple Inc. is looking at buying U.K. startup for $1 billion.
            ```
            Each sentence is a `Span` object.
        *   **Word Tokenization:** The `Doc` object itself is a sequence of `Token` objects. You can iterate over it to get individual tokens.
            ```python
            # Continuing from above
            print("\nTokens in the document:")
            for token in doc:
                print(f"'{token.text}' (Lemma: {token.lemma_}, POS: {token.pos_}, Is Punct: {token.is_punct})")
            # Output example for the first few tokens:
            # 'Apple' (Lemma: Apple, POS: PROPN, Is Punct: False)
            # 'Inc.' (Lemma: Inc., POS: PROPN, Is Punct: False)
            # '.' (Lemma: ., POS: PUNCT, Is Punct: True)
            ```
            Each `token` object has numerous attributes (text, lemma, part-of-speech tag, dependency relations, entity type, etc.).

3.  **NLTK vs. spaCy - A Quick Comparison:**

    | Feature               | NLTK                                       | spaCy                                           |
    |-----------------------|--------------------------------------------|-------------------------------------------------|
    | **Primary Goal**      | Education, Research, Flexibility           | Production, Performance, Ease of Use             |
    | **Approach**          | String-based, modular, many algorithms   | Object-oriented (`Doc`, `Token`), integrated pipelines |
    | **Performance**       | Generally slower                           | Very fast (Cython backend)                       |
    | **Pre-trained Models** | Provides some, but more focused on building blocks | Core strength, provides robust models for many languages |
    | **Learning Curve**    | Easier for individual algorithms; piecing together pipelines can be complex | Steeper for object model initially, then very productive |
    | **Tokenization**      | `word_tokenize`, `sent_tokenize` functions | Part of the `nlp()` pipeline, accessed via `Doc` |
    | **Best For**          | Learning NLP concepts, academic research, experimenting with various low-level components | Building applications, processing large text volumes, when speed and ready-to-use pipelines are key |

    **Key Takeaway:** Both libraries are excellent and serve different primary purposes. NLTK is fantastic for learning how things work under the hood and for accessing a wide array of linguistic resources. spaCy shines when you need to build efficient, production-grade NLP applications quickly. They are not strictly mutually exclusive, though for common tasks, you'll often choose one based on your needs.

## Python Code for Practice:

Please refer to the `nltk_spacy_basics.py` script in this session's directory. It provides hands-on examples for installation verification, sentence tokenization, and word tokenization using both NLTK and spaCy. Make sure you have installed the libraries and necessary models/data before running.

**(Link or reference to the `nltk_spacy_basics.py` from the previous response would go here.)**

## Setup Reminders:

1.  **Install NLTK:** `pip install nltk`
2.  **Download NLTK 'punkt' (at a minimum):**
    ```python
    import nltk
    nltk.download('punkt')
    ```
3.  **Install spaCy:** `pip install spacy`
4.  **Download a spaCy language model (e.g., small English model):**
    `python -m spacy download en_core_web_sm`

## Next Steps:

With a basic understanding of NLTK and spaCy, we are now ready to dive into more specific NLP techniques in Module 2, such as text preprocessing (normalization, stop word removal, stemming, lemmatization), Part-of-Speech tagging, and Named Entity Recognition. We will see how both NLTK and spaCy facilitate these tasks.

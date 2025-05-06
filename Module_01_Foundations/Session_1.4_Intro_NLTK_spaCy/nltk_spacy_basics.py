pip install nltk spacy
python -m spacy download en_core_web_sm

import nltk
nltk.download('punkt')

# Module_01_Foundations/Session_1.4_Intro_NLTK_spaCy/nltk_spacy_basics.py

def run_nltk_practice():
    print("--- NLTK Introduction & Practice ---")
    try:
        import nltk
        print("NLTK imported successfully.")

        # Ensure 'punkt' tokenizer data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' resource not found. Downloading...")
            nltk.download('punkt', quiet=True)
            print("'punkt' downloaded.")
        except Exception as e:
            print(f"An unexpected error occurred checking for 'punkt': {e}")
            return # Exit if critical NLTK data is missing

        # Sample Text
        text_sample = "Dr. Jane Doe is a leading expert in Natural Language Processing. N.L.P. combines AI and linguistics. It's fascinating, isn't it?"

        print(f"\nSample Text: \"{text_sample}\"")

        # 1. Sentence Tokenization with NLTK
        print("\n1. NLTK Sentence Tokenization:")
        try:
            sentences_nltk = nltk.sent_tokenize(text_sample)
            if not sentences_nltk:
                print("  No sentences tokenized by NLTK.")
            for i, sentence in enumerate(sentences_nltk):
                print(f"  NLTK Sentence {i+1}: {sentence}")
        except Exception as e:
            print(f"  Error during NLTK sentence tokenization: {e}")


        # 2. Word Tokenization with NLTK (for the first sentence)
        print("\n2. NLTK Word Tokenization (for the first NLTK-tokenized sentence):")
        if sentences_nltk:
            try:
                first_sentence_nltk = sentences_nltk[0]
                words_nltk = nltk.word_tokenize(first_sentence_nltk)
                print(f"  Words in first NLTK sentence ('{first_sentence_nltk}'): {words_nltk}")
            except Exception as e:
                print(f"  Error during NLTK word tokenization: {e}")
        else:
            print("  No NLTK sentences available for word tokenization.")

    except ImportError:
        print("NLTK library is not installed. Please install it using: pip install nltk")
    except Exception as e:
        print(f"An critical error occurred initializing NLTK: {e}")
    print("-" * 30)


def run_spacy_practice():
    print("\n--- spaCy Introduction & Practice ---")
    try:
        import spacy
        print("spaCy imported successfully.")

        # Load a spaCy language model
        model_name = "en_core_web_sm" # Small English model
        try:
            nlp = spacy.load(model_name)
            print(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            print(f"spaCy model '{model_name}' not found.")
            print(f"Please download it by running: python -m spacy download {model_name}")
            print("If you have downloaded it and still see this, try restarting your environment or check your Python path.")
            return # Exit if model is not available
        except Exception as e:
            print(f"An unexpected error occurred loading spaCy model: {e}")
            return

        # Sample Text (same as NLTK for comparison)
        text_sample = "Dr. Jane Doe is a leading expert in Natural Language Processing. N.L.P. combines AI and linguistics. It's fascinating, isn't it?"
        print(f"\nSample Text: \"{text_sample}\"")

        # Process text with spaCy
        # The 'nlp' object processes the text and creates a 'Doc' object
        # This 'Doc' object contains all the annotations.
        doc_spacy = nlp(text_sample)

        # 1. Sentence Tokenization with spaCy
        # spaCy's sentence segmentation is usually handled by its parser component.
        # Sentences are accessed via `doc.sents` (which is a generator).
        print("\n1. spaCy Sentence Tokenization:")
        sentences_spacy = list(doc_spacy.sents) # Convert generator to list for easy iteration/counting
        if not sentences_spacy:
            print("  No sentences found by spaCy pipeline.")
        for i, sentence_span in enumerate(sentences_spacy): # sentence_span is a spaCy Span object
            print(f"  spaCy Sentence {i+1}: {sentence_span.text}") # .text gets the string content

        # 2. Word Tokenization with spaCy
        # Tokens are directly accessible by iterating over the Doc object.
        # Each item in the iteration is a spaCy Token object.
        print("\n2. spaCy Word Tokenization (for the entire document):")
        spacy_tokens_text = [token.text for token in doc_spacy]
        print(f"  All spaCy tokens (text only): {spacy_tokens_text}")

        print("\n   Exploring attributes of first few spaCy Tokens in the document:")
        for i, token in enumerate(doc_spacy[:10]): # Look at the first 10 tokens
            if i >= 10 : break # safety break, though slice handles it
            print(f"    Token: '{token.text}'")
            print(f"      Lemma: '{token.lemma_}'")          # Base form of the word
            print(f"      POS Tag: '{token.pos_}' ({spacy.explain(token.pos_)})")        # Simple part-of-speech tag
            print(f"      Fine-grained Tag: '{token.tag_}' ({spacy.explain(token.tag_)})")  # Detailed part-of-speech tag
            print(f"      Is Punctuation: {token.is_punct}")
            print(f"      Is Alpha: {token.is_alpha}")       # Consists of alphabetic characters
            print(f"      Is Stop Word: {token.is_stop}")   # Is it a common stop word?

    except ImportError:
        print("spaCy library is not installed. Please install it using: pip install spacy")
    except Exception as e:
        print(f"A critical error occurred during spaCy practice: {e}")
    print("-" * 30)

if __name__ == "__main__":
    print("=== Running NLTK and spaCy Basics Practice ===")
    run_nltk_practice()
    run_spacy_practice()
    print("\nComparison Note: Observe how NLTK and spaCy might tokenize punctuation or contractions differently.")
    print("For example, 'N.L.P.' or 'isn't'. These differences arise from their underlying rules and models.")
    print("\n=== Practice Complete! ===")

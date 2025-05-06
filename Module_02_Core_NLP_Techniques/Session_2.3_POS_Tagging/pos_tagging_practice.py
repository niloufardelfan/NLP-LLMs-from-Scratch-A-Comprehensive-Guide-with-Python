# NLTK for POS Tagging
import nltk
try:
    from nltk.tokenize import word_tokenize
    # Ensure necessary NLTK data is downloaded
    nltk_resources_to_check = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'universal_tagset': 'taggers/universal_tagset' # For mapping to universal tags
    }
    for resource_name, resource_path in nltk_resources_to_check.items():
        try:
            nltk.data.find(resource_path)
        except (nltk.downloader.DownloadError, LookupError):
            print(f"NLTK resource '{resource_name}' for POS tagging not found. Downloading...")
            nltk.download(resource_name, quiet=True)
except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    exit()
except Exception as e:
    print(f"Error importing NLTK components or downloading data: {e}")
    exit()


# spaCy for POS Tagging
import spacy
MODEL_NAME = "en_core_web_sm"
try:
    nlp_spacy = spacy.load(MODEL_NAME)
except OSError:
    print(f"spaCy model '{MODEL_NAME}' not found. Downloading...")
    spacy.cli.download(MODEL_NAME)
    nlp_spacy = spacy.load(MODEL_NAME)
except ImportError:
    print("spaCy not installed. Please install it: pip install spacy")
    exit()
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    exit()

# Sample sentences for POS tagging
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "Apple Inc. is planning to build a new headquarters in California for $5 billion."
sentence3 = "She was reading a fascinating book about artificial intelligence."

sentences = [sentence1, sentence2, sentence3]

print("--- 1. Part-of-Speech (POS) Tagging with NLTK ---")

# NLTK's default POS tagger uses the Penn Treebank tagset.
# You can view details about the tags:
# print("\nNLTK Penn Treebank Tagset (sample):")
# nltk.help.upenn_tagset('NN.*') # Example: Nouns
# nltk.help.upenn_tagset('VB.*') # Example: Verbs

for i, sentence in enumerate(sentences):
    print(f"\nNLTK POS Tagging for Sentence {i+1}: \"{sentence}\"")
    tokens = word_tokenize(sentence)
    tagged_tokens_nltk = nltk.pos_tag(tokens) # Uses Penn Treebank tags by default
    print("  Tokens and Penn Treebank Tags (NLTK):")
    for token, tag in tagged_tokens_nltk:
        print(f"    '{token}': {tag}")

    # NLTK can also map to a simpler Universal Tagset
    tagged_tokens_universal_nltk = nltk.pos_tag(tokens, tagset='universal')
    print("\n  Tokens and Universal Tags (NLTK):")
    for token, tag in tagged_tokens_universal_nltk:
        print(f"    '{token}': {tag}")


print("\n--- 2. Part-of-Speech (POS) Tagging with spaCy ---")

for i, sentence in enumerate(sentences):
    print(f"\nspaCy POS Tagging for Sentence {i+1}: \"{sentence}\"")
    doc = nlp_spacy(sentence) # Process the sentence with spaCy's pipeline

    print("  Token | Coarse POS (pos_) | Fine-grained TAG (tag_) | Explanation (pos_) | Explanation (tag_)")
    print("  " + "-"*100)
    for token in doc:
        coarse_pos = token.pos_
        fine_grained_tag = token.tag_
        explanation_coarse = spacy.explain(coarse_pos)
        explanation_fine = spacy.explain(fine_grained_tag)
        print(f"  '{token.text}' \t| {coarse_pos:<15} | {fine_grained_tag:<20} | {explanation_coarse:<25} | {explanation_fine}")

print("\n--- Key Observations ---")
print("1. NLTK's `nltk.pos_tag()` by default uses the Penn Treebank tagset, which is quite detailed.")
print("2. NLTK can also use the `tagset='universal'` argument for a simpler, more universal set of tags.")
print("3. spaCy provides two main POS attributes for each token:")
print("   - `token.pos_`: A coarse-grained Universal Dependencies (UD) style tag (e.g., NOUN, VERB, ADJ).")
print("   - `token.tag_`: A fine-grained tag, often from the tagset the underlying model was trained on (e.g., Penn Treebank style for many English models).")
print("4. `spacy.explain(tag_string)` is very useful for understanding what a specific tag means.")
print("5. POS tagging is crucial for many downstream tasks, including accurate lemmatization and named entity recognition.")

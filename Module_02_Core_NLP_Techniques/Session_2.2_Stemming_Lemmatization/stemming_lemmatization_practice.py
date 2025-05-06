# NLTK for Stemming and Lemmatization
import nltk
try:
    from nltk.stem import PorterStemmer, SnowballStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet # For POS tagging mapping to WordNet tags
    from nltk.tokenize import word_tokenize

    # Download necessary NLTK data if not already present
    nltk_resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in nltk_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource in ['wordnet', 'omw-1.4'] else f'taggers/{resource}')
        except nltk.downloader.DownloadError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)
        except LookupError: # Handles cases like 'corpora/wordnet.zip' not found
             print(f"NLTK resource '{resource}' not found (LookupError). Downloading...")
             nltk.download(resource, quiet=True)

except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    exit()
except Exception as e:
    print(f"Error importing NLTK components: {e}")
    exit()


# spaCy for Lemmatization
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


# Sample words and sentences for demonstration
words_to_process = ["studies", "studying", "running", "ran", "meeting", "better", "leaves", "automobiles", "corpora", "feet", "wolves", "geese"]
sentence_to_process = "The quick brown foxes are jumping over the lazy dogs and meeting their friends."


print("--- 1. Stemming with NLTK ---")

# a. Porter Stemmer
print("\na. Porter Stemmer:")
porter = PorterStemmer()
stemmed_words_porter = [porter.stem(word) for word in words_to_process]
print(f"Original words: {words_to_process}")
print(f"Porter stems  : {stemmed_words_porter}")

sentence_tokens = word_tokenize(sentence_to_process.lower()) # Tokenize and lowercase
stemmed_sentence_porter = [porter.stem(token) for token in sentence_tokens]
print(f"\nOriginal sentence tokens: {sentence_tokens}")
print(f"Stemmed sentence (Porter): {' '.join(stemmed_sentence_porter)}")

# b. Snowball Stemmer (English)
print("\nb. Snowball Stemmer (English):")
snowball = SnowballStemmer(language='english')
stemmed_words_snowball = [snowball.stem(word) for word in words_to_process]
print(f"Original words: {words_to_process}")
print(f"Snowball stems: {stemmed_words_snowball}")

stemmed_sentence_snowball = [snowball.stem(token) for token in sentence_tokens]
print(f"\nOriginal sentence tokens: {sentence_tokens}")
print(f"Stemmed sentence (Snowball): {' '.join(stemmed_sentence_snowball)}")

print("\nNotice that stems might not be actual words (e.g., 'studi', 'meet').")


print("\n--- 2. Lemmatization with NLTK (WordNetLemmatizer) ---")
lemmatizer = WordNetLemmatizer()

# a. Lemmatization without Part-of-Speech (POS) tags
# If no POS tag is provided, WordNetLemmatizer defaults to noun ('n').
print("\na. Lemmatization without POS tags (defaults to Noun):")
lemmatized_words_no_pos = [lemmatizer.lemmatize(word) for word in words_to_process]
print(f"Original words : {words_to_process}")
print(f"Lemmas (no POS): {lemmatized_words_no_pos}")
# Observe 'running' -> 'running' (as noun), 'better' -> 'better' (as noun)

# b. Lemmatization with Part-of-Speech (POS) tags for better accuracy
# We need a helper function to map NLTK's POS tags to WordNet POS tags.
def nltk_pos_to_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # WordNetLemmatizer defaults to NOUN if None is passed

print("\nb. Lemmatization WITH POS tags:")
# First, get POS tags for the words/tokens
tagged_tokens = nltk.pos_tag(word_tokenize(" ".join(words_to_process))) # Tag the words as if they were in a sentence
# For the sentence:
tagged_sentence_tokens = nltk.pos_tag(sentence_tokens)

print(f"Original words and NLTK POS tags: {tagged_tokens}")

lemmatized_words_with_pos = []
for word, nltk_tag in tagged_tokens:
    wn_tag = nltk_pos_to_wordnet_pos(nltk_tag)
    if wn_tag is None: # If no specific WordNet tag, lemmatize as is (defaulting to noun)
        lemma = lemmatizer.lemmatize(word)
    else:
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    lemmatized_words_with_pos.append(lemma)

print(f"Original words    : {words_to_process}")
print(f"Lemmas (with POS) : {lemmatized_words_with_pos}")
# Observe 'running' -> 'run' (verb), 'better' -> 'good' (adjective, if tagged as JJR)
# The quality of POS tagging heavily influences lemmatization accuracy.

lemmatized_sentence_with_pos_nltk = []
for token, nltk_tag in tagged_sentence_tokens:
    wn_tag = nltk_pos_to_wordnet_pos(nltk_tag)
    if wn_tag is None:
        lemma = lemmatizer.lemmatize(token)
    else:
        lemma = lemmatizer.lemmatize(token, pos=wn_tag)
    lemmatized_sentence_with_pos_nltk.append(lemma)

print(f"\nOriginal sentence tokens: {sentence_tokens}")
print(f"POS tagged sentence tokens (NLTK): {tagged_sentence_tokens}")
print(f"Lemmatized sentence (NLTK with POS): {' '.join(lemmatized_sentence_with_pos_nltk)}")


print("\n--- 3. Lemmatization with spaCy ---")
# spaCy's pipeline inherently performs POS tagging and uses it for lemmatization.
# Process the words (joined as a string to simulate a document)
spacy_doc_words = nlp_spacy(" ".join(words_to_process))
spacy_doc_sentence = nlp_spacy(sentence_to_process) # Process the original sentence

print("\na. Lemmatization of individual words (via spaCy doc):")
lemmatized_words_spacy = [token.lemma_ for token in spacy_doc_words]
original_tokens_spacy = [token.text for token in spacy_doc_words]
print(f"Original tokens (spaCy): {original_tokens_spacy}") # spaCy's tokenization might differ slightly
print(f"Lemmas (spaCy)       : {lemmatized_words_spacy}")
print("   (spaCy Token - Lemma - POS):")
for token in spacy_doc_words:
    print(f"     '{token.text}' - '{token.lemma_}' - '{token.pos_}'")


print("\nb. Lemmatization of a sentence (via spaCy doc):")
original_sentence_tokens_spacy = [token.text for token in spacy_doc_sentence]
lemmatized_sentence_spacy = [token.lemma_ for token in spacy_doc_sentence]

print(f"\nOriginal sentence tokens (spaCy): {original_sentence_tokens_spacy}")
print(f"Lemmatized sentence (spaCy): {' '.join(lemmatized_sentence_spacy)}")
print("   (spaCy Token - Lemma - POS for sentence):")
for token in spacy_doc_sentence:
    print(f"     '{token.text}' - '{token.lemma_}' - '{token.pos_}'")


print("\n--- Comparison Summary ---")
print("Stemming (e.g., Porter): Fast, rule-based, output may not be a real word.")
print("  'studies' -> 'studi', 'running' -> 'run', 'better' -> 'better'")
print("Lemmatization (NLTK w/o POS): Uses WordNet, defaults to noun, output is real word.")
print("  'studies' -> 'study', 'running' -> 'running' (as noun), 'better' -> 'better' (as noun)")
print("Lemmatization (NLTK w/ POS): More accurate, output is real word based on context.")
print("  'studies' (verb) -> 'study', 'running' (verb) -> 'run', 'better' (adj) -> 'good'")
print("Lemmatization (spaCy): Integrated into pipeline, considers POS, efficient, output is real word.")
print("  'studies' -> 'study', 'running' -> 'run', 'better' -> 'good'")
print("Irregular plurals like 'feet' -> 'foot', 'wolves' -> 'wolf', 'geese' -> 'goose' are typically handled well by good lemmatizers but not by stemmers.")

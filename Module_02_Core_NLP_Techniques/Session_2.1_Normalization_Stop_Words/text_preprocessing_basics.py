import string
import re

# NLTK for stop words
import nltk
try:
    from nltk.corpus import stopwords
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("NLTK 'stopwords' resource not found. Downloading...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
except LookupError: # If 'corpora/stopwords' itself not found (older NLTK path issue)
    print("NLTK 'stopwords' resource not found (LookupError). Downloading...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

# spaCy for stop words
import spacy
MODEL_NAME = "en_core_web_sm"
try:
    nlp_spacy = spacy.load(MODEL_NAME)
except OSError:
    print(f"spaCy model '{MODEL_NAME}' not found. Downloading...")
    spacy.cli.download(MODEL_NAME)
    nlp_spacy = spacy.load(MODEL_NAME)


# Sample Text
raw_text = "  Hello World! This is Session 2.1 of our NLP course. We're learning about text NORMALIZATION and removing stop words, ain't it cool?  "
print(f"Original Raw Text:\n'{raw_text}'\n")

# --- 1. Text Normalization ---
print("--- 1. Text Normalization ---")

# a. Case Folding (Lowercase Conversion)
print("\na. Case Folding (Lowercase):")
lower_text = raw_text.lower()
print(f"Lowercased Text:\n'{lower_text}'")

# b. Punctuation Removal
# For this step, we'll typically work with text that's already lowercased.
# We'll also strip leading/trailing whitespace first.
text_for_punct_removal = lower_text.strip()
print(f"\nText for Punctuation Removal (lowercased & stripped):\n'{text_for_punct_removal}'")

# Method 1: Using string.translate() - Generally efficient
# Create a translation table: maps each punctuation char to None (to delete it)
translator = str.maketrans('', '', string.punctuation)
text_no_punct_translate = text_for_punct_removal.translate(translator)
print(f"\nMethod 1: Punctuation Removed (using string.translate()):\n'{text_no_punct_translate}'")

# Method 2: Using Regular Expressions (re.sub())
# Pattern: [^\w\s] matches any character that is NOT a word character (\w) or whitespace (\s)
# This effectively removes punctuation but keeps alphanumeric chars and spaces.
# For more control, you can explicitly list punctuations in the regex.
text_no_punct_regex = re.sub(r"[^\w\s]", '', text_for_punct_removal)
print(f"\nMethod 2: Punctuation Removed (using regex [^\\w\\s]):\n'{text_no_punct_regex}'")

# More specific regex: replace specific punctuation marks with an empty string
# Example: text_specific_punct_regex = re.sub(r"[,.!?]", '', text_for_punct_removal)

# Note: The choice of punctuation removal method can impact tokenization later.
# Often, tokenization is done first, and then punctuation is removed from individual tokens,
# or smart tokenizers handle punctuation. For now, we demonstrate removing from the whole string.
# We'll use the `text_no_punct_translate` for further steps as it's generally robust.
normalized_text_stage1 = text_no_punct_translate # or text_no_punct_regex, depending on preference

print("\n--- 2. Tokenization (Simple Split for now) ---")
# For now, we'll use a simple split. More advanced tokenization comes later.
# It's better to tokenize AFTER major cleaning like punctuation if the tokenizer doesn't handle it well.
# If your tokenizer is smart (like spaCy's), you might do normalization on tokens.
# Here, we tokenize the string from which punctuation was globally removed.
tokens_before_stopwords = normalized_text_stage1.split()
print(f"Tokens (after normalization, before stop word removal):\n{tokens_before_stopwords}")


print("\n--- 3. Stop Word Removal ---")

# a. Using NLTK's Stop Word List
print("\na. Stop Word Removal (NLTK):")
stop_words_nltk = set(stopwords.words('english')) # Use set for faster lookups
# print(f"NLTK English Stop Words (sample): {list(stop_words_nltk)[:15]}")

filtered_tokens_nltk = []
for token in tokens_before_stopwords:
    if token not in stop_words_nltk:
        filtered_tokens_nltk.append(token)
# Using list comprehension:
# filtered_tokens_nltk = [token for token in tokens_before_stopwords if token not in stop_words_nltk]
print(f"Tokens after NLTK stop word removal:\n{filtered_tokens_nltk}")


# b. Using spaCy for Stop Word Removal
# spaCy processes raw text and its tokens have an `is_stop` attribute.
# For a fair comparison, we should give spaCy the same starting point (raw but perhaps lowercased text)
# or process the already tokenized list. Let's process the original raw_text with spaCy for a typical workflow.

print("\nb. Stop Word Removal (spaCy):")
# Process the *original cleaned text* (lowercased) to let spaCy do its own tokenization,
# which is generally more robust than a simple split().
# Let's use the lowercased, stripped text before global punctuation removal to see how spaCy handles it.
spacy_doc = nlp_spacy(lower_text.strip()) # Process with spaCy

print("\n   spaCy's initial tokenization (from lower_text.strip()):")
initial_spacy_tokens = [token.text for token in spacy_doc]
print(f"   {initial_spacy_tokens}")


print("\n   spaCy tokens and their 'is_stop' attribute:")
filtered_tokens_spacy = []
for token in spacy_doc:
    print(f"     Token: '{token.text}', Is Stop: {token.is_stop}, Is Punct: {token.is_punct}")
    # We typically filter out punctuation AND stop words when using spaCy this way
    if not token.is_stop and not token.is_punct:
        filtered_tokens_spacy.append(token.text) # or token.lemma_ for lemmatized output

print(f"\nTokens after spaCy stop word AND punctuation removal (from spaCy pipeline):\n{filtered_tokens_spacy}")

# To compare more directly with the NLTK approach on pre-tokenized words:
# (Less common with spaCy, as its strength is its full pipeline)
print("\n   (Alternative) Checking NLTK-style tokens against spaCy's stop list:")
spacy_stop_words = nlp_spacy.Defaults.stop_words # Get spaCy's stop word set
# print(f"SpaCy English Stop Words (sample): {list(spacy_stop_words)[:15]}")
filtered_tokens_spacy_manual_check = [
    token for token in tokens_before_stopwords if token not in spacy_stop_words
]
print(f"Tokens after checking against spaCy's stop word list (on pre-tokenized input):\n{filtered_tokens_spacy_manual_check}")


print("\n--- Final Processed Text (NLTK example) ---")
final_text_nltk = " ".join(filtered_tokens_nltk)
print(f"Reconstructed text (NLTK stop words removed): '{final_text_nltk}'")

print("\n--- Final Processed Text (spaCy pipeline example) ---")
final_text_spacy = " ".join(filtered_tokens_spacy)
print(f"Reconstructed text (spaCy stop words & punct removed): '{final_text_spacy}'")

print("\nNote: The exact list of stop words can differ between NLTK and spaCy,")
print("and spaCy's tokenization + linguistic features provide a more integrated approach.")

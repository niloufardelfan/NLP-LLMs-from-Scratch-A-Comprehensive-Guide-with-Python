# NLTK for NER
import nltk
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag, ne_chunk

    # Ensure necessary NLTK data is downloaded
    nltk_resources_to_check = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words' # ne_chunker might need this
    }
    for resource_name, resource_path in nltk_resources_to_check.items():
        try:
            nltk.data.find(resource_path)
        except (nltk.downloader.DownloadError, LookupError):
            print(f"NLTK resource '{resource_name}' for NER not found. Downloading...")
            nltk.download(resource_name, quiet=True)

except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    exit()
except Exception as e:
    print(f"Error importing NLTK components or downloading data: {e}")
    exit()

# spaCy for NER and visualization
import spacy
from spacy import displacy # For visualizing NER
MODEL_NAME = "en_core_web_sm" # Using the small model, medium/large might be more accurate
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


# Sample sentences for NER
sentence1 = "Apple Inc., founded by Steve Jobs and Steve Wozniak, is an American multinational technology company headquartered in Cupertino, California."
sentence2 = "Google Maps shows that the Eiffel Tower is located in Paris, France and was completed in March 1889 for $1.5 million."
sentence3 = "The meeting is scheduled for next Tuesday at 3 PM with Dr. Emily Carter from Microsoft."

sentences = [sentence1, sentence2, sentence3]

print("--- 1. Named Entity Recognition (NER) with NLTK ---")
# NLTK's ne_chunk uses POS tags as input.
# The default NLTK NER is trained on the ACE corpus and recognizes limited entity types.

for i, sentence in enumerate(sentences):
    print(f"\nNLTK NER for Sentence {i+1}: \"{sentence}\"")
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    # ne_chunk with binary=False gives typed entities, binary=True just marks NE or not NE
    ner_tree_nltk = nltk.ne_chunk(pos_tags, binary=False)

    print("  NLTK Named Entities (Tree Structure):")
    # To extract entities, you need to traverse the tree.
    # If a node is an nltk.Tree, its label() is the entity type.
    # Its leaves are the (token, POS) tuples.
    extracted_entities_nltk = []
    for subtree in ner_tree_nltk:
        if isinstance(subtree, nltk.Tree): # If it's an entity
            entity_label = subtree.label()
            entity_text_parts = [token for token, pos in subtree.leaves()]
            entity_text = " ".join(entity_text_parts)
            extracted_entities_nltk.append((entity_text, entity_label))
            # print(f"    Entity: '{entity_text}', Label: {entity_label}") # Simple print
        # else: # Not an entity, just a (token, POS) tuple
            # print(f"    Token: '{subtree[0]}', POS: {subtree[1]}") # For debugging tree structure
    
    if extracted_entities_nltk:
        for entity_text, entity_label in extracted_entities_nltk:
            print(f"    Found Entity: '{entity_text}', Type: {entity_label}")
    else:
        print("    No specific entities found by NLTK's default chunker (or only non-entity tokens).")

    # You can try ner_tree_nltk.draw() to visualize if Tkinter is configured (opens a new window)
    # For example: if i == 0: ner_tree_nltk.draw()


print("\n\n--- 2. Named Entity Recognition (NER) with spaCy ---")
# spaCy's models come with pre-trained NER components.

for i, sentence in enumerate(sentences):
    print(f"\nspaCy NER for Sentence {i+1}: \"{sentence}\"")
    doc = nlp_spacy(sentence) # Process the sentence with spaCy's pipeline

    print("  spaCy Named Entities:")
    if not doc.ents: # Check if any entities were found
        print("    No entities found by spaCy in this sentence.")
    for ent in doc.ents:
        entity_text = ent.text
        entity_label = ent.label_
        label_explanation = spacy.explain(entity_label)
        print(f"    Entity: '{entity_text}', Type: {entity_label} ({label_explanation})")
        # print(f"      Start char: {ent.start_char}, End char: {ent.end_char}")


print("\n\n--- 3. Visualizing NER with spaCy's displacy ---")
print("spaCy's displaCy can render NER results visually.")
print("If you are running this in a Jupyter Notebook, set jupyter=True for direct rendering.")
print("Otherwise, displacy.serve() can host it locally, or displacy.render() can return HTML.")

# Example for one sentence:
doc_for_viz = nlp_spacy(sentence1)
html_ner = displacy.render(doc_for_viz, style="ent", page=False) # page=False returns HTML string

# To serve it (opens in browser automatically on http://localhost:5000):
# print("\nAttempting to serve displaCy NER visualization for sentence 1...")
# print("Open your browser to http://localhost:5000 to view.")
# print("Press Ctrl+C in the terminal to stop serving.")
# try:
#     # displacy.serve(doc_for_viz, style="ent") # This will block until stopped
#     print("   (displacy.serve() commented out for non-blocking execution in this script demo)")
#     print("   To try it, uncomment the line above and run the script directly.")
# except KeyboardInterrupt:
#     print("\nDisplacy server stopped by user.")
# except Exception as e:
#     print(f"Could not start displacy server: {e}")

print(f"\nHTML for NER visualization (Sentence 1):\n<!-- START NER HTML -->\n{html_ner}\n<!-- END NER HTML -->")
print("\n(You can save the above HTML content to a .html file and open it in a browser to see the visualization.)")


print("\n--- Key Observations ---")
print("1. NLTK's `ne_chunk` requires POS-tagged input and its default model has limited entity types.")
print("   Extracting entities from NLTK's tree output requires some traversal logic.")
print("2. spaCy integrates NER into its pipeline. Entities are easily accessible via `doc.ents`.")
print("3. spaCy's models (even `en_core_web_sm`) often recognize a broader range of entities and can be more robust.")
print("4. `spacy.explain(label)` is helpful for understanding entity types.")
print("5. `displacy` is a powerful tool for visualizing NER results from spaCy.")

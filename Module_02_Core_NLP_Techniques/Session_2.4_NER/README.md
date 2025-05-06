# Session 2.4: Named Entity Recognition (NER)

Welcome to Session 2.4! In this session, we'll explore **Named Entity Recognition (NER)**, a key information extraction task in NLP. NER systems identify and categorize named entities – real-world objects like persons, organizations, locations, dates, monetary values, etc. – in unstructured text.

## Learning Objectives:

*   Understand what Named Entity Recognition is and its significance in extracting structured information.
*   Identify common types of named entities.
*   Perform NER using NLTK (specifically, its `ne_chunk` functionality).
*   Perform NER using spaCy and explore its entity attributes.
*   Learn how to visualize NER results using spaCy's `displacy` tool.
*   Appreciate the applications of NER in various domains.

## What is Named Entity Recognition (NER)?

Named Entity Recognition (NER), also known as entity chunking, entity identification, or entity extraction, is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories.

**Example:**
Sentence: `Apple Inc., founded by Steve Jobs, is planning to build a new headquarters in Cupertino, California for $5 billion by 2025.`

NER Output could be:
*   `Apple Inc.`: **ORG** (Organization)
*   `Steve Jobs`: **PERSON**
*   `Cupertino`: **GPE** (Geopolitical Entity - typically cities, states, countries)
*   `California`: **GPE**
*   `$5 billion`: **MONEY**
*   `2025`: **DATE**

## Why is NER Important?

NER is crucial for transforming unstructured text into structured data, which can then be used for:

1.  **Information Retrieval:** Improving search accuracy by recognizing entities in queries and documents (e.g., searching for "films starring Tom Hanks").
2.  **Question Answering:** Identifying entities in questions and candidate answers to find relevant information.
3.  **Content Categorization & Recommendation Systems:** Classifying documents or recommending content based on the entities mentioned.
4.  **Knowledge Graph Population:** Extracting entities and their relationships to build knowledge graphs.
5.  **Customer Support:** Automatically extracting product names, customer IDs, or issue types from support tickets.
6.  **Resume Parsing:** Extracting skills, company names, and educational institutions from resumes.
7.  **Medical Informatics:** Identifying diseases, drugs, and patient names in clinical notes (requires specialized models).

## Common Entity Types:

While the exact set of entity types can vary depending on the NER system and its training data, some common ones include:

*   **PERSON:** People's names.
*   **ORG (Organization):** Companies, agencies, institutions.
*   **GPE (Geopolitical Entity):** Countries, cities, states. Often used interchangeably with `LOC` in some contexts or `LOC` might refer to more general locations.
*   **LOC (Location):** Non-GPE locations, mountain ranges, bodies of water.
*   **DATE:** Absolute or relative dates and periods.
*   **TIME:** Times of day.
*   **MONEY:** Monetary values, including currency.
*   **PERCENT:** Percentage values.
*   **FAC (Facility):** Buildings, airports, highways, bridges.
*   **PRODUCT:** Objects, vehicles, foods, etc. (often specific to domain).
*   **EVENT:** Named hurricanes, battles, wars, sports events, etc.
*   **LAW:** Named documents made into laws.
*   **LANGUAGE:** Any named language.
*   **NORP (Nationalities or Religious or Political groups)**

## How NER Works (Brief Overview):

NER systems have evolved:
*   **Rule-Based Systems:** Used hand-crafted rules often based on gazetteers (lists of known entities) and linguistic patterns. Brittle and hard to maintain.
*   **Machine Learning-Based Systems:**
    *   These systems learn from annotated data (text where entities are already labeled).
    *   Features often include words themselves, POS tags, capitalization patterns, surrounding words, etc.
    *   Common algorithms included Hidden Markov Models (HMMs), Maximum Entropy Models (MaxEnt), Conditional Random Fields (CRFs – very popular for sequence labeling tasks like NER).
*   **Deep Learning-Based Systems:**
    *   Modern NER systems typically use neural networks, such as BiLSTMs with a CRF layer on top, or Transformer-based models (e.g., BERT, spaCy's transformers).
    *   These models can learn complex features directly from the data.

## NER with NLTK:

*   NLTK provides a pre-trained NER chunker through `nltk.ne_chunk()`.
*   It typically requires **POS-tagged tokens** as input.
*   The default NLTK NER model is trained on the ACE (Automatic Content Extraction) corpus and recognizes a limited set of entity types like `PERSON`, `ORGANIZATION`, `GPE`, `LOCATION`, `FACILITY`, `DATE`, `TIME`, `MONEY`, `PERCENT`.
*   The output is a tree structure where entities are represented as subtrees.

```python
import nltk
from nltk.tokenize import word_tokenize
# Ensure 'maxent_ne_chunker' and 'words' (for NER model) are downloaded
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

text = "Barack Obama visited Paris with Michelle Obama on September 5th, 2021."
tokens = word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
ner_tree_nltk = nltk.ne_chunk(pos_tags) # Takes POS-tagged tokens
# ner_tree_nltk.draw() # This would open a window to visualize the tree (if Tkinter is set up)

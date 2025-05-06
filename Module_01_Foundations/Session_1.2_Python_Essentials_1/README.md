# Session 1.2: Python Essentials for NLP - Part 1

Welcome to Session 1.2! Now that we have an overview of NLP, it's time to gear up with the fundamental Python tools we'll use throughout this course. This session focuses on core Python data structures and operations essential for text manipulation.

## Learning Objectives:

*   Refresh and solidify understanding of core Python data types: strings, lists, and dictionaries, with a focus on their application to text.
*   Master common string manipulation methods for cleaning, transforming, and analyzing text.
*   Learn how to perform file input/output (I/O) operations to read text data from files and write results.

## Key Concepts Covered:

1.  **Variables and Basic Data Types (Recap):**
    *   **Variables:** Think of variables as labels or names you assign to storage locations in your computer's memory. These locations hold data.
        ```python
        message = "Hello, NLP!"
        count = 10
        pi_value = 3.14159
        is_learning = True
        ```
    *   **Strings (`str`):**
        *   **Definition:** Sequences of characters, enclosed in single (`'...'`), double (`"..."`), or triple (`'''...'''` or `"""..."""`) quotes. Triple quotes are often used for multi-line strings or docstrings.
        *   **NLP Relevance:** The primary way text data is represented and manipulated.
        *   **Immutability:** Once a string is created, it cannot be changed directly. Operations that seem to modify a string actually create a *new* string.
            ```python
            greeting = "hello"
            # greeting = "H"  # This would cause a TypeError
            new_greeting = "H" + greeting[1:] # Creates a new string: "Hello"
            ```
    *   **Lists (`list`):**
        *   **Definition:** Ordered, mutable (changeable) collections of items, enclosed in square brackets `[]`. Items can be of different data types.
        *   **NLP Relevance:** Used to store sequences of words (tokens), sentences, documents, or features.
            ```python
            words = ["natural", "language", "processing", 101]
            words = "is fun" # Lists are mutable
            print(words) # Output: ['natural', 'language', 'processing', 'is fun']
            ```
    *   **Dictionaries (`dict`):**
        *   **Definition:** Unordered (in Python < 3.7, ordered in Python >= 3.7) collections of key-value pairs, enclosed in curly braces `{}`. Each key must be unique and immutable (e.g., string, number, tuple).
        *   **NLP Relevance:** Extremely useful for storing frequencies of words (word counts), vocabularies (word to index mapping), configuration settings, or structured data extracted from text.
            ```python
            word_freq = {"the": 150, "a": 95, "nlp": 25}
            print(word_freq["nlp"]) # Output: 25
            word_freq["learning"] = 10 # Add a new entry
            ```

2.  **String Methods for Text Manipulation:**
    Python strings come with a rich set of built-in methods that are indispensable for NLP. Here are some of the most common ones:

    *   `lower()` / `upper()`:
        *   **Purpose:** Convert string to all lowercase or all uppercase.
        *   **NLP Use:** Essential for text normalization, ensuring consistent casing before analysis (e.g., "Apple" and "apple" are treated as the same word).
            ```python
            text = "NLP is FUN!"
            print(text.lower())  # Output: nlp is fun!
            print(text.upper())  # Output: NLP IS FUN!
            ```
    *   `strip()` / `lstrip()` / `rstrip()`:
        *   **Purpose:** Remove leading and/or trailing whitespace (spaces, tabs, newlines) by default. Can also remove specified characters.
        *   **NLP Use:** Cleaning raw text data which often contains unwanted whitespace.
            ```python
            dirty_text = "  Some extra spaces. \n"
            print(f"'{dirty_text.strip()}'")  # Output: 'Some extra spaces.'
            print(f"'{dirty_text.lstrip()}'") # Output: 'Some extra spaces. \n'
            print(f"'{dirty_text.rstrip()}'") # Output: '  Some extra spaces.'
            custom_strip = "---heading---".strip('-')
            print(custom_strip) # Output: heading
            ```
    *   `split(delimiter=None)`:
        *   **Purpose:** Divides a string into a list of substrings (tokens) based on a specified `delimiter`. If no delimiter is given, it splits by any whitespace and handles multiple spaces effectively.
        *   **NLP Use:** Fundamental for **tokenization** (breaking text into words or sentences).
            ```python
            sentence = "NLP is a fascinating field."
            words = sentence.split() # Default: splits by space
            print(words) # Output: ['NLP', 'is', 'a', 'fascinating', 'field.']

            csv_line = "apple,red,5"
            attributes = csv_line.split(',')
            print(attributes) # Output: ['apple', 'red', '5']
            ```
    *   `join(iterable)`:
        *   **Purpose:** Concatenates elements of an iterable (like a list of strings) into a single string, with the string calling the method used as a separator. It's the inverse of `split()`.
        *   **NLP Use:** Reconstructing text from a list of tokens, or creating formatted strings.
            ```python
            word_list = ["Python", "for", "NLP"]
            joined_string = " ".join(word_list)
            print(joined_string) # Output: Python for NLP

            dashed_date = "-".join(["2024", "05", "06"])
            print(dashed_date) # Output: 2024-05-06
            ```
    *   `find(substring, start=0, end=len(string))` / `index(substring, start=0, end=len(string))`:
        *   **Purpose:** `find()` returns the starting index of the first occurrence of `substring`. Returns -1 if not found. `index()` is similar but raises a `ValueError` if the substring is not found.
        *   **NLP Use:** Locating specific patterns or keywords within text.
            ```python
            text = "The quick brown fox jumps over the lazy dog."
            print(text.find("fox"))    # Output: 16
            print(text.find("cat"))    # Output: -1
            # print(text.index("cat")) # Would raise ValueError
            ```
    *   `replace(old, new, count=-1)`:
        *   **Purpose:** Returns a new string where all occurrences of `old` substring are replaced with `new` substring. The optional `count` argument limits the number of replacements.
        *   **NLP Use:** Text cleaning (e.g., correcting typos, removing unwanted characters), standardization.
            ```python
            text = "I love cats. Cats are great."
            new_text = text.replace("cats", "dogs")
            print(new_text) # Output: I love dogs. Dogs are great.
            limited_replace = text.replace("Cats", "Dogs", 1) # Case-sensitive
            print(limited_replace) # Output: I love cats. Dogs are great.
            ```
    *   `startswith(prefix, start=0, end=len(string))` / `endswith(suffix, start=0, end=len(string))`:
        *   **Purpose:** Checks if the string starts or ends with a specified `prefix` or `suffix`. Returns `True` or `False`.
        *   **NLP Use:** Filtering text (e.g., finding lines that start with a specific character, identifying file types by extension).
            ```python
            filename = "report.txt"
            print(filename.endswith(".txt"))  # Output: True
            print(filename.startswith("doc")) # Output: False
            ```
    *   **f-strings (Formatted String Literals):**
        *   **Purpose:** Introduced in Python 3.6, f-strings provide a concise and convenient way to embed Python expressions inside string literals for formatting. Prepend `f` or `F` to the string.
        *   **NLP Use:** Creating dynamic output messages, logging, formatting text for display.
            ```python
            name = "NLP"
            version = 1.0
            message = f"Welcome to {name} version {version}! The year is {2020 + 4}."
            print(message) # Output: Welcome to NLP version 1.0! The year is 2024.
            ```

3.  **Working with Lists:**
    Lists are versatile for storing sequences of textual elements.
    *   **Creation:** `my_list = ["sentence one", "sentence two", "sentence three"]`
    *   **Indexing & Slicing:**
        *   `my_list[0]` gives the first element.
        *   `my_list[-1]` gives the last element.
        *   `my_list[0:2]` gives a new list containing the first two elements (index 0 and 1).
    *   **Common List Methods for NLP:**
        *   `append(item)`: Adds an item to the end (e.g., adding a new token to a list of tokens).
        *   `extend(iterable)`: Appends all items from another iterable (e.g., combining two lists of sentences).
        *   `insert(index, item)`: Inserts an item at a specific position.
        *   `remove(item)`: Removes the first occurrence of an item.
        *   `pop(index)`: Removes and returns the item at a given index.
        *   `len(list)`: Gets the number of items (e.g., number of words in a sentence, number of sentences in a document).

4.  **Working with Dictionaries:**
    Dictionaries are key for frequency counts and mappings.
    *   **Creation:** `vocab = {"word": 1, "another_word": 2}`
    *   **Accessing:** `vocab["word"]` (raises KeyError if key doesn't exist). `vocab.get("unknown_word", 0)` (returns default value, 0 in this case, if key not found).
    *   **Adding/Updating:** `vocab["new_word"] = 3`
    *   **Iterating:**
        ```python
        for key in vocab:
            print(key, vocab[key])
        for key, value in vocab.items(): # More Pythonic
            print(key, value)
        ```

5.  **File Input/Output (I/O) for Text Data:**
    NLP often involves reading text from files (corpora, datasets) and writing processed text or results back to files.
    *   **The `with open(...) as ...:` statement:** This is the recommended way to work with files. It ensures that the file is properly closed automatically, even if errors occur during processing.
    *   **Modes for Opening Files:**
        *   `'r'`: **Read mode** (default). Opens a file for reading. Raises an error if the file does not exist.
        *   `'w'`: **Write mode**. Opens a file for writing. If the file exists, its content is **truncated (erased)**. If the file does not exist, it is created.
        *   `'a'`: **Append mode**. Opens a file for appending. Data written to the file is added to the end. If the file does not exist, it is created.
        *   `'r+'`: **Read and Write mode**. Opens a file for both reading and writing.
        *   Add `b` for binary mode (e.g., `'rb'`, `'wb'`) for non-text files, but for NLP, we primarily use text modes.
    *   **Reading from Files:**
        ```python
        # Assume 'my_document.txt' exists with some text
        try:
            with open('my_document.txt', 'r', encoding='utf-8') as file: # Always specify encoding
                # 1. Read the entire file content into a single string
                # content = file.read()
                # print(content)

                # 2. Read one line at a time (efficient for large files)
                # for line in file:
                #     print(line.strip()) # .strip() removes leading/trailing whitespace including newline

                # 3. Read all lines into a list of strings
                lines = file.readlines()
                for line in lines:
                    print(line.strip())
        except FileNotFoundError:
            print("Error: The file 'my_document.txt' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        ```
        **Important:** Always specify `encoding='utf-8'` (or the correct encoding of your file) when opening text files to avoid issues with special characters. UTF-8 is a very common and flexible encoding.
    *   **Writing to Files:**
        ```python
        processed_data = ["This is the first processed line.", "And this is the second."]
        try:
            with open('output.txt', 'w', encoding='utf-8') as outfile:
                for line in processed_data:
                    outfile.write(line + "\n") # Manually add newline character
                # Or use: outfile.writelines([line + "\n" for line in processed_data])
            print("Data written to output.txt")
        except Exception as e:
            print(f"An error occurred while writing: {e}")
        ```

## Python Code for Practice:

Please refer to the `practice_python_part1.py` file in this session's directory for hands-on exercises covering all the concepts discussed above. It's highly recommended to run the script, understand its output, and try modifying it to experiment further.

**(Link or reference to the `practice_python_part1.py` from the previous response would go here if this were a live GitHub repo page, e.g., `[Link to Code](./practice_python_part1.py)`)**

## Next Steps:

In the next session, we'll continue exploring Python essentials, focusing on control flow, functions, and a powerful tool for pattern matching in text: Regular Expressions.

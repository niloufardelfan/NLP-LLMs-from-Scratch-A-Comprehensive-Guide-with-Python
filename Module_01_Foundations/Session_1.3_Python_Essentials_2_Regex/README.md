# Session 1.3: Python Essentials for NLP - Part 2 & Introduction to Regular Expressions (Regex)

Welcome to Session 1.3! Building on our Python fundamentals, this session covers control flow structures, how to write reusable code with functions, and introduces Regular Expressions (Regex) – a crucial tool for pattern matching in text.

## Learning Objectives:

*   Utilize Python's control flow statements (`if-elif-else`, `for`, `while`) for text processing logic.
*   Write and use functions to create modular and reusable code for NLP tasks.
*   Understand the basics of Regular Expressions and use Python's `re` module for finding and manipulating text patterns.
*   Appreciate the power of list comprehensions for concise data manipulation.

## Key Concepts Covered:

1.  **Control Flow Statements:**
    Control flow statements dictate the order in which code is executed. They allow your programs to make decisions and repeat actions, which is essential for processing varied text data.

    *   **`if-elif-else` Statements:**
        *   **Purpose:** Execute different blocks of code based on whether certain conditions are true or false.
        *   **Structure:**
            ```python
            if condition1:
                # Code to execute if condition1 is True
            elif condition2: # Optional: "else if"
                # Code to execute if condition1 is False and condition2 is True
            else: # Optional
                # Code to execute if all preceding conditions are False
            ```
        *   **NLP Use:** Checking for specific keywords, categorizing text based on content, conditional preprocessing steps.
            ```python
            document_length = 250
            if document_length > 500:
                category = "long"
            elif document_length > 100:
                category = "medium"
            else:
                category = "short"
            print(f"The document category is: {category}")
            ```

    *   **`for` Loops:**
        *   **Purpose:** Iterate over a sequence (like a list of words, characters in a string, or lines in a file) or other iterable objects.
        *   **Structure:**
            ```python
            for item in iterable:
                # Code to execute for each item
            ```
        *   **NLP Use:** Processing each word in a sentence, each sentence in a document, applying an operation to a collection of texts.
            ```python
            sentences = ["NLP is fun.", "Python is powerful.", "Let's learn!"]
            word_counts = []
            for sentence in sentences:
                words = sentence.split()
                word_counts.append(len(words))
                print(f"Sentence: '{sentence}' has {len(words)} words.")
            print(f"Word counts per sentence: {word_counts}")
            ```

    *   **`while` Loops:**
        *   **Purpose:** Execute a block of code repeatedly as long as a specified condition remains true.
        *   **Structure:**
            ```python
            while condition:
                # Code to execute
                # (Often includes logic to eventually make the condition False)
            ```
        *   **NLP Use:** Less common for simple iteration than `for` loops, but can be used for tasks like iteratively refining results until a criterion is met, or processing input streams.
            ```python
            # Example: Process items until a specific "stop word" is encountered
            data_stream = ["process", "this", "item", "STOP", "dont_process"]
            i = 0
            processed_items = []
            while i < len(data_stream) and data_stream[i] != "STOP":
                processed_items.append(data_stream[i].upper())
                i += 1
            print(f"Processed items before STOP: {processed_items}")
            ```

    *   **`break` and `continue`:**
        *   `break`: Immediately exits the innermost `for` or `while` loop.
        *   `continue`: Skips the rest of the current iteration of the loop and proceeds to the next iteration.

2.  **List Comprehensions:**
    *   **Purpose:** A concise, elegant, and often more efficient way to create lists compared to using a `for` loop with `append()`.
    *   **Structure:** `new_list = [expression for item in iterable if condition]` (the `if condition` part is optional).
    *   **NLP Use:** Transforming lists of tokens (e.g., lowercasing all words), filtering words based on length or type, creating numerical representations from text.
        ```python
        words = ["Apple", "Banana", "Cherry", "Date"]
        # Traditional way to get lowercase words
        lower_words_trad = []
        for word in words:
            lower_words_trad.append(word.lower())

        # Using list comprehension
        lower_words_comp = [word.lower() for word in words]
        print(f"Lowercase words: {lower_words_comp}") # Output: ['apple', 'banana', 'cherry', 'date']

        # With a condition: words longer than 5 characters
        long_words = [word for word in words if len(word) > 5]
        print(f"Long words: {long_words}") # Output: ['Banana', 'Cherry']
        ```

3.  **Functions:**
    *   **Purpose:** Blocks of reusable code that perform a specific task. Functions help organize code, make it more readable, reduce redundancy, and improve maintainability.
    *   **Defining a Function:**
        ```python
        def function_name(parameter1, parameter2, ...): # Parameters are inputs
            """Docstring: Explains what the function does, its parameters, and what it returns."""
            # Body of the function (code to perform the task)
            result = parameter1 + parameter2 # Example operation
            return result # Optional: value returned by the function
        ```
    *   **Calling a Function:** `my_result = function_name(argument1, argument2)` (Arguments are the actual values passed).
    *   **Docstrings:** The triple-quoted string right after the `def` line is a docstring. It's good practice to write clear docstrings. You can access it using `function_name.__doc__`.
    *   **Return Values:** Functions can return one or more values. If no `return` statement is present, or `return` is used without an expression, the function implicitly returns `None`.
    *   **Default Argument Values:** You can provide default values for parameters, making them optional when the function is called.
        ```python
        def preprocess_text(text, lowercase=True, remove_punctuation=False):
            """A simple text preprocessing function."""
            if lowercase:
                text = text.lower()
            if remove_punctuation:
                # (Simplified punctuation removal for example)
                text = text.replace(".", "").replace(",", "")
            return text

        doc1 = "Hello, World."
        print(preprocess_text(doc1)) # Output: hello, world. (lowercase=True by default)
        print(preprocess_text(doc1, remove_punctuation=True)) # Output: hello world
        print(preprocess_text(doc1, lowercase=False)) # Output: Hello, World.
        ```
    *   **NLP Use:** Creating functions for common NLP tasks like tokenizing text, removing stop words, lemmatizing words, calculating TF-IDF, etc.

4.  **Introduction to Regular Expressions (Regex):**
    Regular Expressions (often shortened to "regex" or "regexp") are incredibly powerful sequences of characters that define a search pattern. They are a fundamental tool for sophisticated text processing.

    *   **Why Regex for NLP?**
        *   Extracting structured information (emails, phone numbers, dates, URLs, hashtags).
        *   Advanced tokenization (e.g., splitting by complex delimiters, handling hyphens or apostrophes correctly).
        *   Data cleaning (removing HTML tags, special characters).
        *   Validating text formats.
        *   Finding and replacing specific patterns.

    *   **Python's `re` Module:** Python provides built-in support for regex through the `re` module.
        ```python
        import re
        ```

    *   **Basic Regex Metacharacters & Syntax (A Starting Point):**
        Metacharacters are characters with special meanings in regex.
        *   `.` (Dot): Matches any single character *except* a newline (`\n`). `a.b` matches "aab", "axb", etc.
        *   `^` (Caret): Matches the beginning of the string (or the beginning of a line if `re.MULTILINE` flag is used). `^Hello` matches a string that starts with "Hello".
        *   `$` (Dollar): Matches the end of the string (or the end of a line if `re.MULTILINE` flag is used). `world$` matches a string that ends with "world".
        *   `*` (Asterisk): Matches **zero or more** occurrences of the preceding character or group. `ab*c` matches "ac", "abc", "abbc", "abbbc", etc.
        *   `+` (Plus): Matches **one or more** occurrences of the preceding character or group. `ab+c` matches "abc", "abbc", but NOT "ac".
        *   `?` (Question Mark): Matches **zero or one** occurrence of the preceding character or group (makes it optional). `colou?r` matches "color" and "colour".
        *   `{m}`: Matches exactly `m` occurrences. `a{3}` matches "aaa".
        *   `{m,n}`: Matches between `m` and `n` (inclusive) occurrences. `a{2,4}` matches "aa", "aaa", "aaaa".
        *   `[]` (Character Set/Class): Matches any single character *within* the brackets.
            *   `[aeiou]` matches any lowercase vowel.
            *   `[a-zA-Z]` matches any uppercase or lowercase letter.
            *   `[0-9]` matches any digit.
            *   `[^abc]` (caret inside `[]` means NOT) matches any character *except* 'a', 'b', or 'c'.
        *   `|` (OR / Alternation): Matches either the expression before or after the pipe. `cat|dog` matches "cat" or "dog".
        *   `()` (Grouping):
            1.  Groups parts of a pattern together. `(ab)+` matches "ab", "abab", "ababab".
            2.  **Captures** the matched text within the group. This is useful for extracting parts of a match.
        *   `\` (Backslash / Escape Character):
            1.  Escapes a metacharacter to match its literal meaning. `\.` matches a literal dot, not "any character". `\*` matches a literal asterisk.
            2.  Signals special sequences:
                *   `\d`: Matches any digit. Equivalent to `[0-9]`.
                *   `\D`: Matches any non-digit character. Equivalent to `[^0-9]`.
                *   `\w`: Matches any "word" character (alphanumeric: `[a-zA-Z0-9_]`).
                *   `\W`: Matches any non-word character.
                *   `\s`: Matches any whitespace character (space, tab `\t`, newline `\n`, return `\r`, form feed `\f`, vertical tab `\v`).
                *   `\S`: Matches any non-whitespace character.
                *   `\b`: Matches a **word boundary**. This is a zero-width assertion that matches the position between a word character (`\w`) and a non-word character (`\W`), or at the start/end of the string if the first/last character is a word character. `\bcat\b` matches "cat" as a whole word, not as part of "caterpillar".

    *   **Common `re` Module Functions:**
        *   `re.search(pattern, string, flags=0)`: Scans `string` looking for the *first location* where `pattern` produces a match. Returns a **Match object** if successful, `None` otherwise.
            *   Match objects have methods like `group()` (returns the matched string), `start()` (start index), `end()` (end index).
        *   `re.match(pattern, string, flags=0)`: Tries to apply `pattern` *only at the beginning* of `string`. Returns a Match object if the beginning of the string matches, `None` otherwise.
        *   `re.findall(pattern, string, flags=0)`: Finds *all non-overlapping matches* of `pattern` in `string` and returns them as a **list of strings**. If the pattern contains capturing groups, it returns a list of tuples.
        *   `re.sub(pattern, replacement, string, count=0, flags=0)`: Replaces occurrences of `pattern` in `string` with `replacement`. Returns the modified string. `count` limits the number of substitutions.
        *   `re.split(pattern, string, maxsplit=0, flags=0)`: Splits `string` by occurrences of `pattern`. Returns a list of strings.
        *   `re.compile(pattern, flags=0)`: **Compiles** a regex pattern into a regex object. This is more efficient if you're using the same pattern multiple times, as the compilation step is done only once.
            ```python
            compiled_pattern = re.compile(r"\d+") # Pattern to find one or more digits
            result = compiled_pattern.findall("Call me at 123-456 or 7890.")
            # result is ['123', '456', '7890']
            ```
        *   **Raw Strings (r"...")**: It's highly recommended to use raw strings (e.g., `r"\bword\b"`) when defining regex patterns in Python. This prevents backslashes from being interpreted as Python escape sequences before the regex engine sees them.

## Python Code for Practice:

See the `practice_python_part2_regex.py` file in this session's directory for practical examples and exercises on control flow, functions, and regular expressions. We encourage you to experiment with different patterns and texts!

**(Link or reference to the `practice_python_part2_regex.py` from the previous response would go here.)**

## Further Learning (Regex):

*   **Python `re` module documentation:** [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)
*   **Regex101 ([https://regex101.com/](https://regex101.com/)):** An invaluable online tool for building, testing, and debugging regular expressions with explanations.

## Next Steps:

In the next session, we'll introduce two foundational Python libraries specifically designed for NLP: NLTK and spaCy, and perform some basic NLP operations using them.

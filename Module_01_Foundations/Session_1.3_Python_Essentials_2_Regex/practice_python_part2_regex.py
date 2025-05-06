# Module_01_Foundations/Session_1.3_Python_Essentials_2_Regex/practice_python_part2_regex.py
import re # Import the regular expressions module

def run_control_flow_practice():
    print("--- Control Flow Practice ---")
    # 1. if-elif-else
    text_sample = "The NLP course is insightful and practical."
    if "insightful" in text_sample and "practical" in text_sample:
        print("Sentiment: Very Positive.")
    elif "insightful" in text_sample or "practical" in text_sample:
        print("Sentiment: Positive.")
    elif "bad" in text_sample or "boring" in text_sample:
        print("Sentiment: Negative.")
    else:
        print("Sentiment: Neutral or Undetermined.")

    # 2. for loop
    words = ["natural", "language", "processing", "is", "key"]
    print("\nIterating through words:")
    for word in words:
        if len(word) > 3:
            print(f"  '{word.capitalize()}' (Length > 3)")
        else:
            print(f"  '{word}' (Length <= 3)")

    # Iterating with index using enumerate
    print("\nIterating with enumerate:")
    for index, word in enumerate(words):
        print(f"  Index {index}: {word}")

    # 3. while loop
    print("\nWhile loop example:")
    count = 0
    max_iterations = 4
    while count < max_iterations:
        print(f"  While loop iteration: {count}")
        if count == 1:
            print("    Skipping next step in iteration 1 (using continue conceptual equivalent)")
            # If there were more code here, 'continue' would skip it
        if count == 2:
            print("    Breaking loop at iteration 2")
            break
        count += 1
    else: # Else block for while loop executes if the loop terminated normally (not by break)
        print("  While loop finished without a break.")
    print("-" * 20)


def run_list_comprehensions_practice():
    print("--- List Comprehensions Practice ---")
    numbers = list(range(1, 11)) # Numbers from 1 to 10 [1, 2, ..., 10]

    # 1. Squares of all numbers
    squares = [x**2 for x in numbers]
    print(f"Original numbers: {numbers}")
    print(f"Squares: {squares}")

    # 2. Even numbers from the list
    even_numbers = [x for x in numbers if x % 2 == 0]
    print(f"Even numbers: {even_numbers}")

    # 3. Uppercase words longer than 3 characters
    words = ["text", "analytics", "is", "a", "subfield", "of", "NLP"]
    long_uppercase_words = [word.upper() for word in words if len(word) > 3]
    print(f"Original words: {words}")
    print(f"Long uppercase words: {long_uppercase_words}")
    print("-" * 20)

# --- Functions Practice ---
# This is a function definition itself
def analyze_text_length(text_document, length_threshold=10):
    """
    Analyzes if a text document is considered long or short based on a threshold.

    Args:
        text_document (str): The input text to analyze.
        length_threshold (int, optional): The number of words to consider as the
                                          threshold. Defaults to 10.

    Returns:
        str: A message indicating if the text is "long" or "short".
    """
    if not isinstance(text_document, str):
        return "Error: Input must be a string."
    if not isinstance(length_threshold, int) or length_threshold <= 0:
        return "Error: Threshold must be a positive integer."

    word_count = len(text_document.split())
    if word_count > length_threshold:
        return f"The text is considered 'long' ({word_count} words > threshold {length_threshold})."
    else:
        return f"The text is considered 'short' ({word_count} words <= threshold {length_threshold})."

def run_functions_practice():
    print("--- Functions Practice ---")
    doc1 = "This is a fairly short document."
    doc2 = "This document, on the other hand, contains many more words and is therefore significantly longer."

    print(analyze_text_length(doc1))
    print(analyze_text_length(doc2, length_threshold=15))
    print(analyze_text_length(doc1, length_threshold=5))
    print(analyze_text_length(123)) # Test error handling
    print(f"Docstring for analyze_text_length: {analyze_text_length.__doc__}")
    print("-" * 20)


def run_regex_practice():
    print("--- Regular Expressions (Regex) Practice ---")
    sample_text = "Contact us at info@example.com or support@test.org. Our main office is in New York (NY). Call 123-456-7890 or (987)654-3210. Event date: 2024-12-31. #NLP #Python"

    # 1. re.search(): Find the first occurrence of a pattern
    # Let's find a year (4 digits)
    year_pattern = r"\d{4}" # \d matches a digit, {4} means exactly 4 times
    match_year = re.search(year_pattern, sample_text)
    if match_year:
        print(f"First year found: '{match_year.group()}' at position {match_year.start()}-{match_year.end()}")
    else:
        print("No year pattern found.")

    # 2. re.findall(): Find all email addresses
    # A more robust (but still simplified) email pattern
    email_pattern_str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails_found = re.findall(email_pattern_str, sample_text)
    print(f"Emails found: {emails_found}")

    # 3. re.sub(): Replace all occurrences of phone numbers (simplified)
    # Replace phone numbers with "[PHONE_REDACTED]"
    # This is a simplified pattern for North American numbers
    phone_pattern_str = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    # \(? and \)? make parentheses optional. [-.\s]? makes separator optional.
    redacted_text = re.sub(phone_pattern_str, "[PHONE_REDACTED]", sample_text)
    print(f"\nOriginal text:\n{sample_text}")
    print(f"\nText with redacted phones:\n{redacted_text}")

    # 4. re.split(): Split text based on multiple punctuation marks as delimiters
    text_to_split = "word1.word2,word3;word4 word5"
    split_delimiters_pattern = r"[.,;\s]+" # Split by '.', ',', ';', or one/more whitespace
    split_words = re.split(split_delimiters_pattern, text_to_split)
    print(f"\nOriginal text to split: '{text_to_split}'")
    print(f"Split list: {split_words}") # Note: might produce empty strings if delimiters are at ends

    # 5. Using re.compile() for a frequently used pattern (e.g., hashtags)
    hashtag_pattern_compiled = re.compile(r"#\w+") # '#' followed by one or more word characters
    hashtags = hashtag_pattern_compiled.findall(sample_text)
    print(f"\nHashtags found (using compiled pattern): {hashtags}")

    # 6. Capturing Groups: Extract parts of a date (YYYY-MM-DD)
    date_text = "Event date: 2024-03-15 and Deadline: 2025-01-20"
    date_capture_pattern = r"(\d{4})-(\d{2})-(\d{2})" # Each parenthesis is a capturing group
    date_matches = re.findall(date_capture_pattern, date_text)
    print(f"\nCaptured date parts (list of tuples): {date_matches}")
    if date_matches:
        for year, month, day in date_matches:
            print(f"  Year: {year}, Month: {month}, Day: {day}")

    # Using re.finditer() for more detailed match objects with groups
    print("\nUsing finditer for date parts:")
    for match_obj in re.finditer(date_capture_pattern, date_text):
        print(f"  Full match: {match_obj.group(0)}") # group(0) or group() is the whole match
        print(f"    Year (group 1): {match_obj.group(1)}")
        print(f"    Month (group 2): {match_obj.group(2)}")
        print(f"    Day (group 3): {match_obj.group(3)}")

    print("-" * 20)

if __name__ == "__main__":
    print("=== Running Python Essentials Part 2 & Regex Practice ===")
    run_control_flow_practice()
    run_list_comprehensions_practice()
    run_functions_practice()
    run_regex_practice()
    print("\n=== Practice Complete! ===")

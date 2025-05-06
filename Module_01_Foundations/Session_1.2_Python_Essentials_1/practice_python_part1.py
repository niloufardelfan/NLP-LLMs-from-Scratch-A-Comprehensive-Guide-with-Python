# Module_01_Foundations/Session_1.2_Python_Essentials_1/practice_python_part1.py

def run_string_methods_practice():
    print("--- String Methods Practice ---")
    raw_text = "  Natural Language Processing is REALLY Cool!  \n"
    print(f"Original Text: '{raw_text}'")

    # 1. Convert to lowercase
    lower_text = raw_text.lower()
    print(f"Lowercase: '{lower_text}'")

    # 2. Convert to uppercase
    upper_text = raw_text.upper()
    print(f"Uppercase: '{upper_text}'")

    # 3. Strip leading/trailing whitespace
    stripped_text = raw_text.strip()
    print(f"Stripped: '{stripped_text}'")

    # 4. Split into words (tokens)
    words = stripped_text.split() # Default delimiter is whitespace
    print(f"Split into words: {words}")

    data_string = "item1,item2,item3"
    items = data_string.split(',')
    print(f"Split by ',': {items}")

    # 5. Join words back into a sentence
    joined_sentence = " ".join(words)
    print(f"Joined sentence from words: '{joined_sentence}'")

    date_parts = ["2024", "05", "06"]
    joined_date = "-".join(date_parts)
    print(f"Joined date: '{joined_date}'")

    # 6. Find a substring
    find_index_present = stripped_text.find("Language")
    find_index_absent = stripped_text.find("Python")
    print(f"Index of 'Language' (present): {find_index_present}")
    print(f"Index of 'Python' (absent): {find_index_absent}")

    # 7. Replace a substring
    replaced_text_all = stripped_text.replace("REALLY", "very")
    print(f"Replaced 'REALLY' with 'very': '{replaced_text_all}'")
    text_with_is = "NLP is great. Python is also great."
    replaced_text_once = text_with_is.replace("is", "IS", 1) # Replace only the first occurrence
    print(f"Replaced first 'is': '{replaced_text_once}'")

    # 8. startswith() and endswith()
    filename = "document_final.txt"
    print(f"'{filename}' starts with 'document': {filename.startswith('document')}")
    print(f"'{filename}' ends with '.pdf': {filename.endswith('.pdf')}")
    print(f"'{filename}' ends with '.txt': {filename.endswith('.txt')}")

    # 9. f-strings (Formatted String Literals)
    course_name = "NLP Fundamentals"
    module_number = 1
    score = 95.5
    print(f"Welcome to {course_name}, Module {module_number}. Your score is {score:.1f}.") # .1f for 1 decimal place
    print("-" * 20)

def run_list_methods_practice():
    print("--- List Methods Practice ---")
    fruits = ["apple", "banana", "cherry"]
    print(f"Original fruits list: {fruits}")

    # 1. Append an item
    fruits.append("orange")
    print(f"After append('orange'): {fruits}")

    # 2. Extend with another list
    more_fruits = ["grape", "mango"]
    fruits.extend(more_fruits)
    print(f"After extend with {more_fruits}: {fruits}")

    # 3. Insert an item at a specific position
    fruits.insert(1, "blueberry") # Insert at index 1
    print(f"After insert('blueberry', 1): {fruits}")

    # 4. Remove an item by value (first occurrence)
    try:
        fruits.remove("banana")
        print(f"After remove('banana'): {fruits}")
    except ValueError:
        print("'banana' not found in list for removal.")

    # 5. Pop an item by index (removes and returns)
    if fruits: # Check if list is not empty
        popped_fruit = fruits.pop(2) # Pop item at index 2
        print(f"Popped fruit at index 2: '{popped_fruit}', List now: {fruits}")
        last_fruit = fruits.pop() # Pop last item if no index specified
        print(f"Popped last fruit: '{last_fruit}', List now: {fruits}")

    # 6. Index of an item
    if "cherry" in fruits:
        cherry_index = fruits.index("cherry")
        print(f"Index of 'cherry': {cherry_index}")

    # 7. Count occurrences of an item
    fruits.append("apple") # Add another apple for counting
    apple_count = fruits.count("apple")
    print(f"Current fruits list: {fruits}")
    print(f"Count of 'apple': {apple_count}")

    # 8. Sorting a list
    # fruits.sort() # Sorts in-place
    # print(f"Sorted list (in-place): {fruits}")
    # For mixed types, sorting might fail. Let's use a list of numbers
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    numbers.sort()
    print(f"Sorted numbers (in-place): {numbers}")
    numbers.sort(reverse=True)
    print(f"Sorted numbers (descending): {numbers}")

    # Using sorted() to get a new sorted list
    unsorted_numbers = [50, 20, 80, 10]
    new_sorted_list = sorted(unsorted_numbers)
    print(f"Original numbers: {unsorted_numbers}, New sorted list: {new_sorted_list}")


    # 9. Length of a list
    print(f"Number of fruits remaining: {len(fruits)}")
    print("-" * 20)

def run_dictionary_methods_practice():
    print("--- Dictionary Methods Practice ---")
    word_counts = {"hello": 5, "world": 3, "nlp": 10, "python": 8}
    print(f"Original word counts: {word_counts}")

    # 1. Accessing a value by key
    print(f"Count of 'nlp': {word_counts['nlp']}")
    # print(f"Count of 'java': {word_counts['java']}") # This would raise a KeyError

    # 2. Safely accessing a value using get()
    print(f"Count of 'java' (using get with default): {word_counts.get('java', 0)}")
    print(f"Count of 'hello' (using get): {word_counts.get('hello', 0)}")

    # 3. Adding or updating an item
    word_counts["learning"] = 15  # Add new item
    word_counts["nlp"] = 12       # Update existing item
    print(f"Updated word counts: {word_counts}")

    # 4. Removing an item using pop()
    popped_value = word_counts.pop("world") # Removes "world" and returns its value
    print(f"Popped value for 'world': {popped_value}, Dictionary now: {word_counts}")
    # popped_non_existent = word_counts.pop("non_existent_key", "Not Found") # Using default for pop
    # print(f"Trying to pop non-existent key: {popped_non_existent}")

    # 5. Getting keys, values, and items
    print(f"Keys: {list(word_counts.keys())}")
    print(f"Values: {list(word_counts.values())}")
    print(f"Items (key-value pairs): {list(word_counts.items())}")

    # 6. Checking for key existence
    print(f"Is 'python' a key? {'python' in word_counts}")
    print(f"Is 'java' a key? {'java' in word_counts}")
    print("-" * 20)

def run_file_io_practice():
    print("--- File I/O Practice ---")
    output_filename = "sample_output.txt"
    input_filename_for_demo = "sample_input_for_read_demo.txt" # We'll create this

    # 1. Writing to a file
    lines_to_write = [
        "This is the first line written by Python for our NLP course.\n",
        "Python makes file I/O straightforward.\n",
        "Remember to close your files, or use 'with open()'.\n"
    ]
    try:
        with open(output_filename, "w", encoding="utf-8") as f_out: # 'w' for write mode
            for line in lines_to_write:
                f_out.write(line)
            # Alternative: f_out.writelines(lines_to_write)
        print(f"Successfully wrote initial content to '{output_filename}'")

        # Append mode 'a'
        with open(output_filename, "a", encoding="utf-8") as f_append: # 'a' for append mode
            f_append.write("This line was appended later.\n")
        print(f"Successfully appended a line to '{output_filename}'")

    except IOError as e:
        print(f"Error during file writing: {e}")

    # Prepare a file for reading demonstration
    try:
        with open(input_filename_for_demo, "w", encoding="utf-8") as f_temp:
            f_temp.write("Hello from the input demo file.\n")
            f_temp.write("This is the second line.\n")
            f_temp.write("Welcome to NLP text processing.\n")
        print(f"Created '{input_filename_for_demo}' for reading demo.")
    except IOError as e:
        print(f"Error creating demo input file: {e}")


    # 2. Reading from a file
    try:
        print(f"\nReading from '{input_filename_for_demo}':")
        with open(input_filename_for_demo, "r", encoding="utf-8") as f_in: # 'r' for read mode
            # a. Read entire content as one string
            # content = f_in.read()
            # print(f"\nEntire content read with read():\n'''{content}'''")
            # Note: After read(), the file cursor is at the end. For other read methods, reopen or seek(0).

        # Reopen for other read methods for clarity in demo
        with open(input_filename_for_demo, "r", encoding="utf-8") as f_in:
            # b. Read line by line (memory efficient for large files)
            print(f"\nReading line by line:")
            lines_read_one_by_one = []
            for line in f_in: # Iterating over the file object itself
                processed_line = line.strip() # strip() removes leading/trailing whitespace including '\n'
                print(f"  Line read: '{processed_line}'")
                lines_read_one_by_one.append(processed_line)
            print(f"Lines stored from line-by-line reading: {lines_read_one_by_one}")

        with open(input_filename_for_demo, "r", encoding="utf-8") as f_in:
            # c. Read all lines into a list of strings
            all_lines_list = f_in.readlines() # Each string in the list will retain its newline character
            print(f"\nAll lines from readlines(): {all_lines_list}")
            processed_lines_from_readlines = [line.strip() for line in all_lines_list]
            print(f"Processed lines from readlines(): {processed_lines_from_readlines}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename_for_demo}' was not found for reading.")
    except IOError as e:
        print(f"Error during file reading: {e}")
    print("-" * 20)

if __name__ == "__main__":
    print("=== Running Python Essentials Part 1 Practice ===")
    run_string_methods_practice()
    run_list_methods_practice()
    run_dictionary_methods_practice()
    run_file_io_practice()
    print("\n=== Practice Complete! ===")

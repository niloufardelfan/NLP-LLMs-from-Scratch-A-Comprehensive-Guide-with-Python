# Session 3.3: Introduction to Text Classification & Scikit-learn

Welcome to Session 3.3! Now that we can convert text into numerical features (using BoW, TF-IDF, and N-grams), we're ready to apply machine learning algorithms to solve NLP tasks. This session introduces one of the most common NLP tasks: **Text Classification**, and shows how to perform it using the powerful **Scikit-learn** library.

## Learning Objectives:

*   Understand the task of text classification and identify its common applications.
*   Become familiar with the standard machine learning workflow and Scikit-learn's core API:
    *   `fit()`: For training a model.
    *   `transform()`: For transforming data (e.g., by vectorizers).
    *   `fit_transform()`: Combines fitting and transforming.
    *   `predict()`: For making predictions on new data.
*   Learn how to prepare data for a classification task, separating features (X) and labels (y).
*   Understand the importance of splitting data into training and testing sets (`train_test_split`).
*   Train a simple classifier (Multinomial Naive Bayes) on text data for a classification task.
*   Make basic predictions on the test set.

## What is Text Classification?

Text classification (also known as text categorization) is the task of assigning a predefined category or label to a piece of text (a document, sentence, or paragraph). It's a **supervised machine learning** problem, meaning we need a labeled dataset where each text sample is already associated with its correct category.

**Common Applications:**
*   **Spam Detection:** Classifying emails as "spam" or "not spam" (ham).
*   **Sentiment Analysis:** Classifying reviews, tweets, or comments as "positive," "negative," or "neutral."
*   **Topic Labeling:** Assigning news articles to topics like "sports," "politics," "technology."
*   **Language Identification:** Determining the language of a given text.
*   **Intent Recognition:** In chatbots, classifying user utterances into predefined intents (e.g., "book_flight," "check_weather").
*   **Urgency Detection:** Classifying support tickets based on urgency (e.g., "high," "medium," "low").

## The Standard Machine Learning Workflow for Text Classification:

1.  **Data Collection:** Gather a corpus of text documents along with their corresponding labels.
2.  **Text Preprocessing:** Clean the text data (e.g., lowercasing, punctuation removal, stop word removal, stemming/lemmatization - as covered in Module 2).
3.  **Feature Extraction/Vectorization:** Convert the preprocessed text into numerical feature vectors (e.g., using BoW or TF-IDF - covered in Session 3.1 & 3.2). This results in your feature matrix `X`.
4.  **Label Preparation:** Convert your categorical labels into a numerical format if necessary (e.g., "spam" -> 1, "ham" -> 0). This results in your label vector `y`.
5.  **Splitting Data:** Divide the dataset into a **training set** and a **testing set**.
    *   **Training Set:** Used to train the machine learning model (i.e., the model learns patterns from this data).
    *   **Testing Set:** Used to evaluate the performance of the trained model on unseen data. This helps assess how well the model generalizes.
    *   Scikit-learn's `train_test_split` function is commonly used for this.
6.  **Model Selection:** Choose an appropriate classification algorithm (e.g., Naive Bayes, Logistic Regression, SVM).
7.  **Model Training:** "Fit" the selected model to the training data (`X_train`, `y_train`). The model learns the relationship between the features and the labels.
8.  **Model Evaluation:** Use the trained model to make predictions on the test set (`X_test`) and compare these predictions to the actual labels (`y_test`). Calculate performance metrics (accuracy, precision, recall, F1-score - covered in next session).
9.  **Model Deployment/Prediction (Optional):** If the model performs well, it can be deployed to make predictions on new, unlabeled text data.

## Scikit-learn: The Go-To Library for Machine Learning in Python

Scikit-learn (`sklearn`) is a comprehensive and user-friendly library that provides tools for various machine learning tasks, including classification, regression, clustering, dimensionality reduction, model selection, and preprocessing.

**Key API Concepts (Estimator Interface):**
Scikit-learn objects generally follow a consistent API:
*   **Estimator Objects:** Any object that can learn from data. This includes classifiers, regressors, and transformers (like `CountVectorizer`, `TfidfVectorizer`).
*   **`fit(X, y)`:** For supervised learning estimators (like classifiers), this method trains the model using features `X` and labels `y`. For unsupervised estimators or transformers, it might just be `fit(X)`.
*   **`transform(X)`:** For transformers, this method applies the learned transformation (e.g., vectorization) to the data `X`.
*   **`fit_transform(X, [y])`:** A convenience method that combines `fit()` and `transform()` in one step (often more efficient).
*   **`predict(X)`:** For classifiers and regressors, this method makes predictions on new data `X` after the model has been trained.
*   **Parameters:** Estimators are initialized with parameters (hyperparameters) that control their behavior (e.g., `C` in SVM, `alpha` in Naive Bayes).

## Preparing Data for Classification:

*   **Features (`X`):** This will be the numerical matrix produced by your text vectorizer (e.g., `TfidfVectorizer`). Each row represents a document, and each column represents a feature (a word or N-gram from the vocabulary).
*   **Labels (`y`):** This will be a 1D array or list containing the category label for each corresponding document in `X`. Labels should typically be numerical (e.g., 0, 1, 2 for different classes). If your labels are strings (e.g., "spam", "ham"), you'll need to convert them. `LabelEncoder` from Scikit-learn can help, or simple mapping.

## Multinomial Naive Bayes: A Simple and Effective Classifier for Text

*   **Naive Bayes Classifiers:** A family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
*   **Multinomial Naive Bayes (MNB):** Particularly well-suited for classification with discrete features (like word counts from BoW or TF-IDF values, though TF-IDF is technically continuous, MNB often works well).
*   **How it works (simplified for text):** It calculates the probability of a document belonging to a class based on the probability of the words in that document appearing in documents of that class in the training data.
*   **Advantages:**
    *   Computationally efficient and fast to train.
    *   Requires a relatively small amount of training data.
    *   Often performs surprisingly well on text classification tasks, especially as a baseline.
*   **Disadvantages:**
    *   The "naive" independence assumption (that features/words are independent of each other given the class) is often violated in real-world text. However, it still tends to work well in practice.

## Python Code for Practice:

The Python script `text_classification_naive_bayes.py` in this session's directory will demonstrate:
1.  Creating a very simple, small labeled dataset (e.g., positive/negative sentences).
2.  Preprocessing the text (minimal for this example).
3.  Using `TfidfVectorizer` for feature extraction.
4.  Splitting the data into training and testing sets using `train_test_split`.
5.  Training a `MultinomialNB` classifier.
6.  Making predictions on the test set and seeing the raw predictions.

**(Link or reference to the `text_classification_naive_bayes.py` would go here.)**

## Next Steps:

While we've trained a model and made predictions, we haven't formally evaluated its performance. In the next session, we'll explore other common classifiers for text and dive into crucial **Evaluation Metrics** (like accuracy, precision, recall, F1-score) and the **Confusion Matrix** to understand how well our classifier is doing.

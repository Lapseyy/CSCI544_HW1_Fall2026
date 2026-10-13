"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
This is an unimplemented student starter, not a solution or an autograder.
"""
import csv
import html
import re
import contractions
import nltk
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
# For lemmatization/preprocessing
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
# lemmatizer and stop_words are global variables, initialized once and reused across function calls
# Creating once at module level and reused for calls to preprocess_review is more efficient than reinitializing in every function call.
lemmatizer = WordNetLemmatizer()
# filled in on the first call to preprocess_review
# Not importing stopwords from nltk.corpus at the top level to avoid unnecessary downloads or imports if not needed
# Per assigbnment instructions, we are not downloading resources at runtime, so we assume nltk stopwords are already available
stop_words = None  

def prepare_data(path, n_per_class=100000):
    """Return (train_df, test_df, stats), as specified in Section 1/interface."""
    # Read the TSV file, keep only the review and rating columns, and rename them.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        on_bad_lines="skip",
        usecols=["review_body", "star_rating"],
    ).rename(columns={"review_body": "review", "star_rating": "rating"})

    # Convert the rating column to numeric, filter out invalid ratings, and count excluded rows.
    numeric_ratings = pd.to_numeric(df["rating"], errors="coerce")
    
    # valid_mask = numeric_ratings.isin([1, 2, 3, 4, 5])
    # excluded_rows = int((~valid_mask).sum())
    # df = df.loc[valid_mask].copy()
    # df["rating"] = numeric_ratings.loc[valid_mask].astype(int)

    # Valid = rating is an integer 1-5 AND review is not missing/blank
    numeric_ratings = pd.to_numeric(df["rating"], errors="coerce")
    valid_mask = numeric_ratings.isin([1, 2, 3, 4, 5]) & (df["review"].str.strip() != "")
    excluded_rows = int((~valid_mask).sum())

    df = df.loc[valid_mask].copy()
    df["rating"] = numeric_ratings.loc[valid_mask].astype(int)
    df = df[["review", "rating"]]

    rating_counts = (
        df["rating"]
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .astype(int)
        .to_dict()
    )

    stats = {
        "rating_counts": rating_counts,
        "class_counts": {
            "positive": int(df["rating"].isin([4, 5]).sum()),
            "negative": int(df["rating"].isin([1, 2]).sum()),
            "neutral": int((df["rating"] == 3).sum()),
        },
        "excluded_rows": excluded_rows,
    }

    positive = df.loc[df["rating"].isin([4, 5])].copy()
    positive["label"] = 1
    negative = df.loc[df["rating"].isin([1, 2])].copy()
    negative["label"] = 0

    sampled = pd.concat(
        [
            positive.sample(n=n_per_class, replace=False, random_state=42),
            negative.sample(n=n_per_class, replace=False, random_state=42),
        ],
        ignore_index=True,
    )
    train_df, test_df = train_test_split(
        sampled,
        test_size=0.2,
        stratify=sampled["label"],
        random_state=42,
    )
    return train_df, test_df, stats


def clean_review(text):
    """Return one cleaned string, including the empty string when applicable."""
    # s = html.unescape(text)
    # s = s.lower()
    # s = s.replace(r"<[^>]*>", "")  # Remove HTML tags
    # s = s.replace(r"https?://\S+|www\.\S+", "")  # Remove URLs
    # s = s.replace("‘", "'").replace("’", "'")
    # s = contractions.fix(s)
    # s = s.replace(r"[^a-z0-9\s]", " ")  # Remove non-alphanumeric characters
    # s = " ".join(s.split())  # Normalize whitespace
    # s = text
    s = html.unescape(text)
    s = s.lower()
    s = re.sub(r"<[^>]*>", " ", s)                  # tags -> space
    
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)    # URLs -> space
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = contractions.fix(s)
    s = re.sub(r"[^a-z\s]", " ", s)                 # non a-z -> space
    return " ".join(s.split())                       # collapse + strip


def preprocess_review(text):
    """Return stopword-filtered, noun-lemmatized text from cleaned input."""
    global stop_words
    if stop_words is None:
        stop_words = set(nltk_stopwords.words("english")) - set(["no", "not", "nor"])

    saved_text = []
    for word in text.split():
        if word not in stop_words:
            saved_text.append(lemmatizer.lemmatize(word, pos="n"))  # n is for noun lemma

    return " ".join(saved_text)

def extract_features(train_texts, test_texts):
    """Return (fitted_vectorizer, sparse_X_train, sparse_X_test)."""
    vectorizer = tfidf()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

def train_model(name, X_train, y_train):
    """Return a fitted classifier for one of the three specified model names."""
    if name == "Perceptron":
        model = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
    elif name == "Logistic Regression":
        model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000, tol=1e-4)
    elif name == "Naive Bayes":
        model = MultinomialNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown model name: {name}")
    # Fit the model to the training data
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Return unrounded accuracy, precision, recall, f1 in a dictionary."""
    y_pred = model.predict(X)
    # dictionary of the predictions
    
    return {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, average="binary", pos_label=1, zero_division=0),
    "recall": recall_score(y, y_pred, average="binary", pos_label=1, zero_division=0),
    "f1": f1_score(y, y_pred, average="binary", pos_label=1, zero_division=0),
    }
    

def fit_bigram(train_texts, min_count=3):
    """Return a dictionary with vocab, bigrams, contexts; omit zero counts."""
    # prepare sequence and counts
    def to_sequence(text, vocab):
        words = []
        # loops run 0 times if empty text, so we get only <s> and </s> in that edgecase
        for word in text.split():
            if word in vocab:
                words.append(word)
            else:
                words.append("<unk>")

        return ["<s>"] + words + ["</s>"]

    # Count words and filter by min_count
    word_counts = Counter(word for review in train_texts for word in review.split())
    # Create vocabulary with words meeting min_count and add special tokens
    vocab = {word for word, count in word_counts.items() if count >= min_count}
    # Add special tokens to the vocabulary
    vocab.update({"<unk>", "</s>"})
    # Create bigram and context counts
    bigrams = Counter()
    # Create context counts
    contexts = Counter()
    # Loop through each review, convert to sequence, and count bigrams and contexts
    for review in train_texts:
        sequence = to_sequence(review, vocab)
        # 
        for context, word in zip(sequence, sequence[1:]):
            bigrams[(context, word)] += 1
            contexts[context] += 1
    # Return the results as a dictionary
    return {"vocab": vocab, "bigrams": dict(bigrams), "contexts": dict(contexts)}


def evaluate_bigram(model, test_texts, k=1.0):
    """Return a dictionary with unseen, n_tokens, add-k perplexity; do not refit."""
    # Base case k must be greater than 0 to avoid division by zero
    if k <= 0:
        raise ValueError("k must be greater than 0 for add-k smoothing")

    # Extract vocab, bigrams, and contexts from the model
    vocab = model["vocab"]
    bigrams = model["bigrams"]
    contexts = model["contexts"]
    vocab_size = len(vocab)

    unseen = 0
    n_tokens = 0
    log = 0.0

    # Loop through each review in the test set, convert to sequence, and calculate probabilities
    for review in test_texts:
        words = [
            word if word in vocab else "<unk>"
            for word in review.split()
        ]

        # Create a sequence with start and end tokens
        sequence = ["<s>", *words, "</s>"]

        # Loop through each bigram in the sequence and calculate probabilities
        for context, word in zip(sequence, sequence[1:]):
            count = bigrams.get((context, word), 0)
            context_count = contexts.get(context, 0)

            if count == 0:
                unseen += 1

            probability = (count + k) / (context_count + k * vocab_size)
            log += math.log(probability)
            n_tokens += 1
    # Check if n_tokens is zero to avoid division by zero in perplexity calculation
    if n_tokens == 0:
        raise ValueError("test_texts must contain at least one token to compute perplexity")

    return {
        "unseen": unseen,
        "n_tokens": n_tokens,
        "perplexity": math.exp(-log / n_tokens),
    }


def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""
    ##################Section 1: data######################
    train_df, test_df, stats = prepare_data(path, n_per_class)
    #classcounts
    classcounts = stats["class_counts"]
    print(f"Positive reviews: {classcounts['positive']}")
    print(f"Negative reviews: {classcounts['negative']}")
    print(f"Neutral reviews: {classcounts['neutral']}")

    ##################Section 2 (. Data and features)##################
    train_raw = train_df["review"].tolist()
    test_raw = test_df["review"].tolist()
    train_clean = [clean_review(t) for t in train_raw]
    test_clean = [clean_review(t) for t in test_raw]
    print(f"Average length before cleaning: {sum(len(t) for t in train_raw) / len(train_raw):.4f}")
    print(f"Average length after cleaning: {sum(len(t) for t in train_clean) / len(train_clean):.4f}")

    ##################Section 3##################
    train_pre = [preprocess_review(t) for t in train_clean]
    test_pre = [preprocess_review(t) for t in test_clean]
    print(f"Average length before preprocessing: {sum(len(t) for t in train_clean) / len(train_clean):.4f}")
    print(f"Average length after preprocessing: {sum(len(t) for t in train_pre) / len(train_pre):.4f}")

    ##################Section 4##################
    vectorizer, X_train, X_test = extract_features(train_pre, test_pre)
    print(f"TF-IDF matrix shape: {X_train.shape}")

    ##################Section 5, 6, & 7 Classifier metrics (24 lines) ##################
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # Train each model and evaluate on both training and testing sets
    for name in ["Perceptron", "Logistic Regression", "Naive Bayes"]:
        model = train_model(name, X_train, y_train)

        # Evaluate on both training and testing sets
        for split, X, y in [("Training", X_train, y_train), ("Testing", X_test, y_test)]:
            m = evaluate_model(model, X, y)
            print(f"{name} {split} Accuracy: {m['accuracy']:.4f}")
            print(f"{name} {split} Precision: {m['precision']:.4f}")
            print(f"{name} {split} Recall: {m['recall']:.4f}")
            print(f"{name} {split} F1-score: {m['f1']:.4f}")

    ##################Section 8 Bigram model and smoothing comparison (5 lines)##################
    train_series = pd.Series(train_clean, index=train_df.index)
    test_series = pd.Series(test_clean, index=test_df.index)

    lm_train = train_series[train_series != ""].sample(
        n=lm_train_size, replace=False, random_state=42).tolist()
    lm_test = test_series[test_series != ""].sample(
        n=lm_test_size, replace=False, random_state=42).tolist()


    bigram = fit_bigram(lm_train)
    print(f"Bigram vocabulary size: {len(bigram['vocab'])}")

    r = evaluate_bigram(bigram, lm_test, k=1.0)
    print(f"Bigram unseen test transitions: {r['unseen']}")
    print(f"Bigram test perplexity: {r['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.1): {evaluate_bigram(bigram, lm_test, k=0.1)['perplexity']:.4f}")
    print(f"Bigram test perplexity (k=0.01): {evaluate_bigram(bigram, lm_test, k=0.01)['perplexity']:.4f}")


if __name__ == "__main__":
    run_experiment()

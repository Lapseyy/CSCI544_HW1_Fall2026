"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
This is an unimplemented student starter, not a solution or an autograder.
"""
import pandas as pd

def prepare_data(path, n_per_class=100000):
    """Return (train_df, test_df, stats), as specified in Section 1/interface."""
    raise NotImplementedError


def clean_review(text):
    """Return one cleaned string, including the empty string when applicable."""
    raise NotImplementedError


def preprocess_review(text):
    """Return stopword-filtered, noun-lemmatized text from cleaned input."""
    raise NotImplementedError


def extract_features(train_texts, test_texts):
    """Return (fitted_vectorizer, sparse_X_train, sparse_X_test)."""
    raise NotImplementedError


def train_model(name, X_train, y_train):
    """Return a fitted classifier for one of the three specified model names."""
    raise NotImplementedError


def evaluate_model(model, X, y):
    """Return unrounded accuracy, precision, recall, f1 in a dictionary."""
    raise NotImplementedError


def fit_bigram(train_texts, min_count=3):
    """Return a dictionary with vocab, bigrams, contexts; omit zero counts."""
    raise NotImplementedError


def evaluate_bigram(model, test_texts, k=1.0):
    """Return a dictionary with unseen, n_tokens, add-k perplexity; do not refit."""
    raise NotImplementedError


def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""
    raise NotImplementedError


if __name__ == "__main__":
    run_experiment()

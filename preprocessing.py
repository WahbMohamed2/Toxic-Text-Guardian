"""
NLP Preprocessing using NLTK
Demonstrates: Tokenization, Stop Words Removal, POS Tagging, Lemmatization, and Stemming
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import re
import pandas as pd

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("omw-1.4", quiet=True)
print("NLTK data downloaded successfully!")


class NLPPreprocessor:
    """
    NLP Preprocessing Pipeline using NLTK

    Steps:
    1. Tokenization
    2. Stop Words Removal
    3. POS Tagging
    4. Lemmatization (or Stemming)
    """

    def __init__(self, use_stemming=False, remove_stopwords=True):
        """
        Initialize preprocessor

        Args:
            use_stemming (bool): Use stemming instead of lemmatization
            remove_stopwords (bool): Remove stop words
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        # Initialize NLTK tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        """Clean text by removing URLs, special characters, etc."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        """
        Tokenization: Split text into words using NLTK

        Args:
            text (str): Input text

        Returns:
            list: List of tokens
        """
        return word_tokenize(text)

    def remove_stop_words(self, tokens):
        """
        Remove stop words from tokens

        Args:
            tokens (list): List of tokens

        Returns:
            list: Tokens without stop words
        """
        return [token for token in tokens if token not in self.stop_words]

    def pos_tagging(self, tokens):
        """
        POS Tagging: Tag each token with its part of speech

        Args:
            tokens (list): List of tokens

        Returns:
            list: List of (token, POS_tag) tuples
        """
        return pos_tag(tokens)

    def apply_stemming(self, tokens):
        """
        Stemming: Reduce words to their root form using Porter Stemmer

        Args:
            tokens (list): List of tokens

        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def apply_lemmatization(self, tokens):
        """
        Lemmatization: Reduce words to their base form using WordNet Lemmatizer

        Args:
            tokens (list): List of tokens

        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text, verbose=False):
        """
        Complete preprocessing pipeline

        Args:
            text (str): Input text
            verbose (bool): Print intermediate steps

        Returns:
            str: Preprocessed text
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("NLP PREPROCESSING STEPS")
            print(f"{'=' * 60}")
            print(f"Original text: {text}\n")

        # Step 1: Clean text
        cleaned = self.clean_text(text)
        if verbose:
            print(f"After cleaning: {cleaned}\n")

        # Step 2: Tokenization
        tokens = self.tokenize(cleaned)
        if verbose:
            print(f"After tokenization: {tokens}\n")

        # Step 3: Remove stop words (optional)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
            if verbose:
                print(f"After removing stop words: {tokens}\n")

        # Step 4: POS Tagging (for demonstration)
        pos_tags = self.pos_tagging(tokens)
        if verbose:
            print(f"POS Tags: {pos_tags[:10]}...\n")

        # Step 5: Stemming or Lemmatization
        if self.use_stemming:
            tokens = self.apply_stemming(tokens)
            if verbose:
                print(f"After stemming: {tokens}\n")
        else:
            tokens = self.apply_lemmatization(tokens)
            if verbose:
                print(f"After lemmatization: {tokens}\n")

        # Filter out short tokens and non-alphabetic tokens
        tokens = [token for token in tokens if len(token) > 2 and token.isalpha()]

        if verbose:
            print(f"Final processed tokens: {tokens}")
            print(f"{'=' * 60}\n")

        return " ".join(tokens)

    def fit_transform(self, texts, verbose=False):
        """
        Preprocess multiple texts

        Args:
            texts (list or pd.Series): List of texts to preprocess
            verbose (bool): Show progress

        Returns:
            list: List of preprocessed texts
        """
        if verbose:
            print(f"Preprocessing {len(texts)} texts...")

        processed = []
        for i, text in enumerate(texts):
            processed.append(self.preprocess(text, verbose=False))
            if verbose and (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")

        if verbose:
            print(f"Preprocessing complete!")
            print(
                f"Average tokens per text: {sum(len(t.split()) for t in processed) / len(processed):.2f}"
            )

        return processed


def demonstrate_preprocessing():
    """Demonstrate the preprocessing pipeline with examples"""

    # Example text
    sample_text = "How do terrorist organizations fund and execute attacks on foreign soil without being detected?"

    print("\n" + "=" * 70)
    print("DEMONSTRATION: NLP PREPROCESSING WITH NLTK")
    print("=" * 70)

    # Using Lemmatization
    print("\n--- Using Lemmatization ---")
    preprocessor_lemma = NLPPreprocessor(use_stemming=False, remove_stopwords=True)
    processed_lemma = preprocessor_lemma.preprocess(sample_text, verbose=True)

    # Using Stemming
    print("\n--- Using Stemming ---")
    preprocessor_stem = NLPPreprocessor(use_stemming=True, remove_stopwords=True)
    processed_stem = preprocessor_stem.preprocess(sample_text, verbose=True)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Lemmatization: {processed_lemma}")
    print(f"Stemming:      {processed_stem}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demonstrate_preprocessing()
